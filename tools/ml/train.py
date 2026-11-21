"""Train crux's denoiser on training bundles (Noise2Noise).

    python tools/ml/train.py --bundles C:/crux-work/train --test-camera STL \
        --out .local/ml/runs/first

Sets whose camera contains a --test-camera string (or whose bundle folder
contains a --test-bundle string) are held out entirely:
the score on them is how well the network handles an optical train it has
never seen. 10% of the other patches are held back for validation too.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

import bundles as B
from model import FWHM_SCALE, UNet, parameters


def split_pairs(pairs: list[B.Pair], test_cameras: list[str], val: float, seed: int, test_bundles: list[str] = ()):
    """(train, val, test): test = held-out cameras; val = a share of the
    other pairs' patches, chosen once."""
    rng = random.Random(seed)
    held = lambda p: any(t.lower() in p.camera.lower() for t in test_cameras) or any(t.lower() in p.bundle.lower() for t in test_bundles)
    test = [p for p in pairs if held(p)]
    rest = [p for p in pairs if not held(p)]
    train, valid = [], []
    for p in rest:
        idx = list(range(len(p.a)))
        rng.shuffle(idx)
        nv = max(1, int(round(len(idx) * val))) if len(idx) >= 4 else 0
        v, t = sorted(idx[:nv]), sorted(idx[nv:])
        if t:
            train.append(B.Pair(**{**p.__dict__, "a": p.a[t], "b": p.b[t]}))
        if v:
            valid.append(B.Pair(**{**p.__dict__, "a": p.a[v], "b": p.b[v]}))
    return train, valid, test


class Stream(torch.utils.data.IterableDataset):
    """Random training samples. With `cache`, the patches are written once
    to a flat float32 file there and workers memory-map it: Windows spawns
    workers by pickling the dataset, and past ~2 GB of patches (5,973
    pairs' worth, run r8) the pipe write fails."""

    def __init__(self, pairs: list[B.Pair], aug: B.Aug, seed: int, cache: Path | None = None):
        self.pairs, self.aug, self.seed = pairs, aug, seed
        self.weights = [len(p.a) for p in pairs]
        self.cache, self.layout = None, []
        if cache is not None:
            total = sum(p.a.size + p.b.size for p in pairs)
            mm = np.lib.format.open_memmap(cache, mode="w+", dtype=np.float32, shape=(total,))
            pos = 0
            for p in pairs:
                ent = []
                for arr in (p.a, p.b):
                    mm[pos : pos + arr.size] = arr.ravel()
                    ent.append((pos, arr.shape))
                    pos += arr.size
                self.layout.append(ent)
            mm.flush()
            del mm
            self.cache = cache

    def __getstate__(self):
        st = dict(self.__dict__)
        if self.cache is not None:
            st["pairs"] = [B.Pair(**{**p.__dict__, "a": None, "b": None}) for p in self.pairs]
        return st

    def _attach(self):
        if self.cache is None or self.pairs[0].a is not None:
            return
        mm = np.load(self.cache, mmap_mode="r")
        for p, ((pa, sa), (pb, sb)) in zip(self.pairs, self.layout):
            p.a = mm[pa : pa + int(np.prod(sa))].reshape(sa)
            p.b = mm[pb : pb + int(np.prod(sb))].reshape(sb)

    def __iter__(self):
        self._attach()
        info = torch.utils.data.get_worker_info()
        rng = random.Random(self.seed + (info.id if info else 0) * 7919 + int(time.time()))
        while True:
            p = rng.choices(self.pairs, self.weights)[0]
            x, y, fwhm = B.sample(p, rng.randrange(len(p.a)), self.aug, rng)
            f = np.full_like(x, fwhm / FWHM_SCALE)
            yield np.stack([x, f]), y[None]


def denoise(net: torch.nn.Module, a: np.ndarray, sky: float, sigma: float, fwhm: float, device) -> np.ndarray:
    """One patch (sides a multiple of 16), stored values in and out."""
    x = B.forward(a, sky, sigma)
    inp = torch.from_numpy(np.stack([x, np.full_like(x, fwhm / FWHM_SCALE)])[None]).to(device)
    with torch.no_grad():
        y = net(inp)[0, 0].float().cpu().numpy()
    return B.inverse(y, sky, sigma)


def score(pairs: list[B.Pair], fn) -> dict:
    """Noise2Noise score. A and B are independent, so E|A−B|² = V_A + V_B,
    split between them in proportion to their background noise² (which
    also covers star and nebula shot noise, as long as both stacks scale
    alike). The true error of any estimate D of the clean image is then
    E|D−B|² − V_B, and A's own error is V_A. gain = A's error ÷ D's (×1 =
    no better than A; a gain of 4 is worth 4× the frames). 'faint' = pixels
    within 3σ of their patch's median, where the noise shows most."""
    tot = {"all": [0.0, 0.0], "faint": [0.0, 0.0]}
    for p in pairs:
        share_b = p.noise_b**2 / (p.noise_a**2 + p.noise_b**2)
        for i in range(len(p.a)):
            a, b = p.a[i], p.b[i]
            d = fn(p, a)
            ab, db = (a - b) ** 2, (d - b) ** 2
            faint = np.abs(a - np.median(a)) < 3 * p.noise_a
            for k, m in (("all", slice(None)), ("faint", faint)):
                e = float(ab[m].mean())
                tot[k][0] += e * (1 - share_b)
                tot[k][1] += float(db[m].mean()) - e * share_b
    out = {}
    for k, (ea, ed) in tot.items():
        out[k] = ea / ed if ed > 0 else float("inf")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundles", nargs="+", required=True, type=Path)
    ap.add_argument("--test-camera", nargs="*", default=[])
    ap.add_argument("--test-bundle", nargs="*", default=[], help="hold out bundles whose folder name contains one of these")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--crop", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--no-aug", action="store_true")
    ap.add_argument("--offset-match", action="store_true", help="shift each target patch's median onto the input's")
    ap.add_argument("--scale-aug", action="store_true", help="downscale by a random factor in [1, 2] (half the time) instead of exactly 2× (a quarter of the time)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pairs = B.load_all(args.bundles)
    train, valid, test = split_pairs(pairs, args.test_camera, args.val, args.seed, args.test_bundle)
    desc = lambda ps: sorted({f"{p.camera} · {p.object} · {p.filter}" for p in ps})
    print(f"device {device}; {sum(len(p.a) for p in train)} training patches from {len(train)} pairs, "
          f"{sum(len(p.a) for p in valid)} validation, {sum(len(p.a) for p in test)} held out")
    for d in desc(test):
        print(f"  held out: {d}")
    if not train:
        raise SystemExit("nothing to train on")

    aug = B.Aug(crop=args.crop) if not args.no_aug else B.Aug(crop=args.crop, p_bin=0, p_blur=0, p_bayer=0)
    aug.offset_match = args.offset_match
    if args.scale_aug:
        aug.continuous, aug.p_bin = True, 0.5
    args.out.mkdir(parents=True, exist_ok=True)
    cache = args.out / "train_patches.npy" if args.workers > 0 else None
    loader = torch.utils.data.DataLoader(Stream(train, aug, args.seed, cache), batch_size=args.batch,
                                         num_workers=args.workers, persistent_workers=args.workers > 0)
    net = UNet(base=args.base, depth=args.depth).to(device)
    print(f"U-Net, {parameters(net) / 1e6:.2f} M parameters")
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.iters, pct_start=0.05)
    args.out.mkdir(parents=True, exist_ok=True)
    log = open(args.out / "log.csv", "w", newline="")
    w = csv.writer(log)
    w.writerow(["iter", "loss", "val_all", "val_faint", "test_all", "test_faint", "seconds"])

    def evaluate():
        net.eval()
        fn = lambda p, a: denoise(net, a, p.sky, p.noise_a, p.fwhm, device)
        v = score(valid, fn) if valid else {}
        t = score(test, fn) if test else {}
        net.train()
        return v, t

    t0 = time.time()
    run, it = 0.0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        loss = torch.nn.functional.l1_loss(net(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        run = 0.98 * run + 0.02 * loss.item() if it else loss.item()
        it += 1
        if it % 1000 == 0 or it == args.iters:
            v, t = evaluate()
            el = time.time() - t0
            print(f"{it:6d}  loss {run:.4f}  val gain {v.get('all', math.nan):.2f} (faint {v.get('faint', math.nan):.2f})"
                  f"  held-out gain {t.get('all', math.nan):.2f} (faint {t.get('faint', math.nan):.2f})  {el:.0f} s", flush=True)
            w.writerow([it, run, v.get("all"), v.get("faint"), t.get("all"), t.get("faint"), round(el)])
            log.flush()
            torch.save({"model": net.state_dict(), "base": args.base, "depth": args.depth, "iter": it}, args.out / "model.pt")
        if it >= args.iters:
            break

    # Baselines on the same patches, for scale.
    def blurred(s):
        return lambda p, a: B.blur(a, s)

    report = {"iters": it, "parameters": parameters(net), "train": desc(train), "held_out": desc(test)}
    v, t = evaluate()
    report["network"] = {"val": v, "held_out": t}
    for s in (0.7, 1.0):
        report[f"gaussian_{s}"] = {"val": score(valid, blurred(s)) if valid else {}, "held_out": score(test, blurred(s)) if test else {}}
    print(json.dumps(report, indent=1))
    export(net.cpu().eval(), args.out, report)
    # The patch cache is only for the workers (several GB).
    del loader
    if cache is not None:
        try:
            cache.unlink()
        except OSError as e:
            print(f"left {cache}: {e}")


def finite(v):
    """JSON has no NaN or infinity: write them as null."""
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    if isinstance(v, dict):
        return {k: finite(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [finite(x) for x in v]
    return v


def export(net: torch.nn.Module, out: Path, report: dict):
    x = torch.zeros(1, 2, 64, 64)
    torch.onnx.export(net, x, out / "denoise.onnx", input_names=["input"], output_names=["output"],
                      dynamic_axes={"input": {2: "h", 3: "w"}, "output": {2: "h", 3: "w"}}, opset_version=17,
                      dynamo=False)
    meta = {"format": "crux-denoise", "version": 1, "k": B.K, "fwhm_scale": FWHM_SCALE, "multiple": 16,
            "inputs": ["asinh((v - sky) / (k * noise))", "fwhm / fwhm_scale"], **report}
    (out / "denoise.json").write_text(json.dumps(finite(meta), indent=1, allow_nan=False), encoding="utf-8")
    print(f"exported {out / 'denoise.onnx'}")


if __name__ == "__main__":
    main()
