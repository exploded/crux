"""Train crux's star corrector / sharpener (deconvolution network).

Truths are the centres of crux's own masters (round stars), denoised with
crux's own denoiser. Each training input is a truth patch blurred by a
random aberration kernel (elongation, a coma tail, both, or none), plus real
stack noise (half-stack differences from the bundles). The network learns
to give back the truth: stars corrected, noise removed.

Inputs: [asinh image, FWHM / 4, sharpen]. With sharpen = 0 kernels are
aberrations only (shape corrected, size kept); with sharpen = 1 they also
carry symmetric blur, so the network also tightens stars.

Never uses any other program's output (see README).

    python deconv.py --masters C:/crux-work/train/masters --bundles C:/crux-work/train \
        --denoiser ../../.local/ml/runs/r2-holdout-stl-qhy9/model.pt --out ../../.local/ml/runs/d1
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import bundles as B
from model import FWHM_SCALE, UNet, parameters
from train import finite

KS = 25  # kernel side
CROP = 128
CENTRE = 0.7  # share of each master (per axis) used for truths
MAX_SCALE, P_SCALE = 2.0, 0.6  # downscale truths by up to this, this often
FLUX_W = 0.0  # weight of the flux-conservation term


# ----------------------------------------------------------------- truths


def denoise_full(net, img: np.ndarray, sky: float, sigma: float, fwhm: float, device, tile: int = 1024, ov: int = 32) -> np.ndarray:
    """Tiled denoise of a whole image with the denoiser network."""
    h, w = img.shape
    out = np.zeros_like(img)
    wsum = np.zeros_like(img)
    step = tile - 2 * ov
    ys = list(range(0, max(1, h - tile), step)) + [max(0, h - tile)]
    xs = list(range(0, max(1, w - tile), step)) + [max(0, w - tile)]
    ramp = np.ones(tile, np.float32)
    r = 0.5 - 0.5 * np.cos(np.pi * (np.arange(ov) + 0.5) / ov)
    ramp[:ov], ramp[-ov:] = r, r[::-1]
    for y in ys:
        for x in xs:
            t = img[y : y + tile, x : x + tile]
            th, tw = t.shape
            ph, pw = (-th) % 16, (-tw) % 16
            tp = np.pad(t, ((0, ph), (0, pw)), mode="reflect")
            xin = B.forward(tp, sky, sigma)
            inp = torch.from_numpy(np.stack([xin, np.full_like(xin, fwhm / FWHM_SCALE)])[None]).to(device)
            with torch.no_grad():
                d = B.inverse(net(inp)[0, 0].float().cpu().numpy(), sky, sigma)[:th, :tw]
            wt = np.outer(ramp[:th] if y else np.ones(th), ramp[:tw] if x else np.ones(tw)).astype(np.float32)
            out[y : y + th, x : x + tw] += d * wt
            wsum[y : y + th, x : x + tw] += wt
    return out / np.maximum(wsum, 1e-6)


def measure_fwhm(img: np.ndarray, sky: float, sigma: float, r: int = 8) -> float:
    """Median second-moment FWHM of isolated bright unsaturated stars."""
    from numpy.lib.stride_tricks import sliding_window_view as sw

    win = sw(np.pad(img, 3, mode="edge"), (7, 7)).max((-1, -2))
    peaks = np.argwhere((img == win) & (img - sky > 50 * sigma) & (img < 0.9))
    yy, xx = np.mgrid[-r : r + 1, -r : r + 1]
    out = []
    for y, x in peaks[:: max(1, len(peaks) // 400)]:
        if y < r or x < r or y >= img.shape[0] - r or x >= img.shape[1] - r:
            continue
        c = img[y - r : y + r + 1, x - r : x + r + 1] - sky
        c = np.clip(c - np.median(c[[0, -1], :]), 0, None)
        s = c.sum()
        if s <= 0:
            continue
        mx, my = (c * xx).sum() / s, (c * yy).sum() / s
        v = ((c * ((xx - mx) ** 2 + (yy - my) ** 2)).sum() / s) / 2
        if v > 0:
            out.append(2.3548 * math.sqrt(v))
    return float(np.median(out)) if out else 3.0


class Truth:
    def __init__(self, name, img, sky, sigma, fwhm):
        self.name, self.img, self.sky, self.sigma, self.fwhm = name, img, sky, sigma, fwhm


def crux_fwhm(workspaces: list[Path]) -> dict[str, float]:
    """Each master's FWHM as crux measures it (weighted over its frames,
    from crux-integration.json), keyed by the master's file stem. The Rust
    side feeds crux's own measurement, so training must use the same."""
    out = {}
    for j in (j for w in workspaces for j in Path(w).glob("*/crux-integration.json")):
        for r in json.loads(j.read_text(encoding="utf-8")).values():
            if r.get("fwhm"):
                out[Path(r["out"]).stem] = float(r["fwhm"])
    return out


def load_truths(folder: Path, denoiser: Path, cache: Path, device, workspaces: list[Path] | None = None) -> list[Truth]:
    cache.mkdir(parents=True, exist_ok=True)
    known = crux_fwhm(workspaces or [Path(folder).parent.parent])
    ck = torch.load(denoiser, map_location=device)
    net = UNet(base=ck.get("base", 32), depth=ck.get("depth", 3)).to(device).eval()
    net.load_state_dict(ck["model"])
    out = []
    for f in sorted(Path(folder).glob("*.fits")):
        c = cache / (f.stem + ".npz")
        stem = f.stem.split("_", 1)[1] if not f.stem.startswith("masterLight") else f.stem
        if c.is_file() and c.stat().st_mtime > f.stat().st_mtime:
            z = np.load(c)
            out.append(Truth(f.stem, z["img"], float(z["sky"]), float(z["sigma"]), known.get(stem, float(z["fwhm"]))))
            continue
        _, a = B.read_fits_cube(f)
        a = a[0]
        h, w = a.shape
        y0, x0 = int(h * (1 - CENTRE) / 2), int(w * (1 - CENTRE) / 2)
        a = np.ascontiguousarray(a[y0 : h - y0, x0 : w - x0])
        sky = float(np.median(a[::7, ::7]))
        sigma = B.noise_of(a[::1, ::1][:2048, :2048])
        fw0 = measure_fwhm(a, sky, sigma)
        t0 = time.time()
        d = denoise_full(net, a, sky, sigma, fw0, device)
        fwhm = measure_fwhm(d, sky, sigma)
        np.savez(c, img=d.astype(np.float32), sky=sky, sigma=sigma, fwhm=fwhm)
        print(f"  {f.stem}: {a.shape[1]}×{a.shape[0]}, noise {sigma:.2e}, FWHM {fw0:.2f} → {fwhm:.2f} px (denoised in {time.time() - t0:.0f} s)", flush=True)
        out.append(Truth(f.stem, d, sky, sigma, known.get(stem, fwhm)))
    return out


def noise_bank(roots: list[Path], exclude: list[str]) -> list[np.ndarray]:
    """Unit-σ real stack noise: (A − B) / √(σA² + σB²) from bundle pairs."""
    bank = []
    for p in B.load_all(roots):
        if any(e.lower() in p.bundle.lower() for e in exclude):
            continue
        s = math.sqrt(p.noise_a**2 + p.noise_b**2)
        for i in range(0, len(p.a), 3):
            d = (p.a[i] - p.b[i]) / s
            # Stars and nebula cancel; clip what's left of bright residues.
            bank.append(np.clip(d, -6, 6).astype(np.float32))
    return bank


# ---------------------------------------------------------------- kernels


def kernel(rng: random.Random, sharpen: bool) -> np.ndarray:
    """A random PSF error: elongation, a coma tail, both, or none, peak at
    the centre; with `sharpen`, also a symmetric Gaussian blur."""
    r = KS // 2
    yy, xx = np.mgrid[-r : r + 1, -r : r + 1].astype(np.float32)
    k = np.zeros((KS, KS), np.float32)
    kind = rng.random()
    th = rng.uniform(0, math.pi)
    c, s = math.cos(th), math.sin(th)
    u, v = xx * c + yy * s, -xx * s + yy * c
    if kind < 0.15:
        k[r, r] = 1.0
    else:
        elong = kind < 0.75 and (kind < 0.45 or rng.random() < 0.5)
        coma = kind >= 0.45
        if elong:
            sm, sn = rng.uniform(0.6, 2.6), 0.35
        else:
            sm, sn = 0.35, 0.35
        base = np.exp(-0.5 * ((u / sm) ** 2 + (v / sn) ** 2))
        if coma:
            L = rng.uniform(0.8, 3.5)
            fan = rng.uniform(0.0, 0.35)
            tail = np.zeros_like(k)
            for t in np.linspace(0.5, 3 * L, 24):
                wt = math.exp(-t / L)
                sw = 0.45 + fan * t
                tail += wt * np.exp(-0.5 * (((u - t) / 0.5) ** 2 + (v / sw) ** 2)) / sw
            tail *= rng.uniform(0.3, 1.2) * base.sum() / max(tail.sum(), 1e-9)
            k = base + tail
        else:
            k = base
    if sharpen:
        sb = rng.uniform(0.5, 1.8)
        g = np.exp(-0.5 * (xx**2 + yy**2) / sb**2)
        k = F.conv2d(torch.from_numpy(k)[None, None], torch.from_numpy(g / g.sum())[None, None], padding=r)[0, 0].numpy()
    return (k / k.sum()).astype(np.float32)


def kernel_fwhm(k: np.ndarray) -> float:
    r = KS // 2
    yy, xx = np.mgrid[-r : r + 1, -r : r + 1]
    s = k.sum()
    mx, my = (k * xx).sum() / s, (k * yy).sum() / s
    v = (k * ((xx - mx) ** 2 + (yy - my) ** 2)).sum() / s / 2
    return 2.3548 * math.sqrt(max(v, 0.0))


# ------------------------------------------------------------------- data


# Cameras of the masters other than James's ASI2600, by file-name prefix.
OTHER_CAMERAS = ("ASI294MC", "EOS80D", "Origin")


def camera_of(name: str) -> str:
    p = name.split("_", 1)[0]
    return p if p in OTHER_CAMERAS else "ASI2600"


class Picker:
    """Draws a truth: uniformly, or (with `share`) the ASI2600 that share of
    the time and the other cameras the rest equally, uniform within each."""

    def __init__(self, truths, share=None):
        self.truths, self.share = truths, share
        self.by = {}
        for t in truths:
            self.by.setdefault(camera_of(t.name), []).append(t)

    def __call__(self, rng):
        if self.share is None or len(self.by) < 2 or "ASI2600" not in self.by:
            return rng.choice(self.truths)
        if rng.random() < self.share:
            return rng.choice(self.by["ASI2600"])
        return rng.choice(self.by[rng.choice([c for c in self.by if c != "ASI2600"])])


def batch(truths, bank, n, rng, device, test_scale=None):
    pick = truths if callable(truths) else Picker(truths)
    xs, ys = [], []
    for _ in range(n):
        t = pick(rng)
        f = test_scale if test_scale else (rng.uniform(1.0, MAX_SCALE) if rng.random() < P_SCALE else 1.0)
        big = CROP + KS - 1
        c = int(math.ceil(big * f)) + 2
        h, w = t.img.shape
        y, x = rng.randrange(h - c), rng.randrange(w - c)
        patch = torch.from_numpy(np.ascontiguousarray(t.img[y : y + c, x : x + c]))[None, None].to(device)
        patch = F.adaptive_avg_pool2d(patch, (big, big)) if f > 1.0 else patch[..., :big, :big]
        if rng.random() < 0.5:
            patch = patch.flip(-1)
        patch = torch.rot90(patch, rng.randrange(4), (-2, -1))
        sharpen = rng.random() < 0.5
        k = kernel(rng, sharpen)
        blurred = F.conv2d(patch, torch.from_numpy(k)[None, None].to(device))
        truth = patch[..., KS // 2 : KS // 2 + CROP, KS // 2 : KS // 2 + CROP]
        sigma = t.sigma * rng.uniform(0.3, 1.5)
        nz = rng.choice(bank)
        ny, nx = rng.randrange(nz.shape[0] - CROP + 1), rng.randrange(nz.shape[1] - CROP + 1)
        noise = torch.from_numpy(np.rot90(nz[ny : ny + CROP, nx : nx + CROP], rng.randrange(4)).copy())[None, None].to(device)
        inp = blurred + noise * sigma
        fw_t = t.fwhm / f
        fw_in = math.hypot(fw_t, kernel_fwhm(k))
        kk = 3.0 * sigma
        xi = torch.asinh((inp - t.sky) / kk)
        yt = torch.asinh((truth - t.sky) / kk)
        xs.append(torch.cat([xi, torch.full_like(xi, fw_in / FWHM_SCALE), torch.full_like(xi, float(sharpen))], 1))
        ys.append(yt)
    return torch.cat(xs), torch.cat(ys)


def flux_loss(out: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Light kept: 8 × 8 block sums of output and truth in linear units
    (noise σ: v / σ = 3 sinh(x)), compared relative to the signal there,
    so stars neither gain nor lose flux."""
    o = F.avg_pool2d(3.0 * torch.sinh(out.clamp(-12, 12)), 8)
    t = F.avg_pool2d(3.0 * torch.sinh(y.clamp(-12, 12)), 8)
    return ((o - t).abs() / (t.abs() + 3.0)).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--masters", type=Path, required=True)
    ap.add_argument("--workspaces", type=Path, nargs="*", default=None, help="where crux-integration.json gives each master's FWHM (default: the masters folder's grandparent)")
    ap.add_argument("--bundles", type=Path, nargs="+", required=True)
    ap.add_argument("--exclude", nargs="*", default=["NGC 6188", "QHY9"], help="bundles left out of the noise bank")
    ap.add_argument("--denoiser", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--iters", type=int, default=15000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-scale", type=float, default=2.0)
    ap.add_argument("--p-scale", type=float, default=0.6)
    ap.add_argument("--flux-weight", type=float, default=0.0)
    ap.add_argument("--camera-share", type=float, default=None, help="draw ASI2600 truths this share of the time, other cameras the rest equally (default: every truth equally)")
    ap.add_argument("--skip-truths", nargs="*", default=[], help="leave out masters whose file name contains one of these")
    args = ap.parse_args()
    global MAX_SCALE, P_SCALE, FLUX_W
    MAX_SCALE, P_SCALE, FLUX_W = args.max_scale, args.p_scale, args.flux_weight
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    print("truths:", flush=True)
    # Truths are denoised by --denoiser, so each denoiser gets its own cache
    # (the first runs' cache, made with r2, keeps its old name).
    tag = args.denoiser.parent.name
    cache = args.out.parent / ("deconv_cache" if tag == "r2-holdout-stl-qhy9" else f"deconv_cache_{tag}")
    truths = load_truths(args.masters, args.denoiser, cache, device, args.workspaces)
    truths = [t for t in truths if not any(k in t.name for k in args.skip_truths)]
    bank = noise_bank(args.bundles, args.exclude)
    print(f"{len(truths)} truths, {len(bank)} noise patches; FWHM (crux) " + ", ".join(f"{t.fwhm:.1f}" for t in truths), flush=True)
    # A fixed validation set from two truths' worth of draws.
    vrng = random.Random(999)
    pick = Picker(truths, args.camera_share)
    print("cameras: " + ", ".join(f"{c} {len(v)}" for c, v in sorted(pick.by.items())) + (f"; ASI2600 share {args.camera_share}" if args.camera_share is not None else ""), flush=True)
    val = [batch(pick, bank, args.batch, vrng, device) for _ in range(8)]
    net = UNet(base=args.base, depth=args.depth, inputs=3).to(device)
    print(f"U-Net, {parameters(net) / 1e6:.2f} M parameters", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.iters, pct_start=0.05)
    args.out.mkdir(parents=True, exist_ok=True)
    t0, run = time.time(), 0.0
    for it in range(1, args.iters + 1):
        x, y = batch(pick, bank, args.batch, rng, device)
        out = net(x)
        loss = F.l1_loss(out, y)
        if FLUX_W > 0:
            loss = loss + FLUX_W * flux_loss(out, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        run = 0.98 * run + 0.02 * loss.item() if it > 1 else loss.item()
        if it % 1000 == 0 or it == args.iters:
            net.eval()
            with torch.no_grad():
                vl = sum(F.l1_loss(net(a), b).item() for a, b in val) / len(val)
                vi = sum(F.l1_loss(a[:, :1], b).item() for a, b in val) / len(val)
            net.train()
            print(f"{it:6d}  loss {run:.4f}  val {vl:.4f} (input as is: {vi:.4f})  {time.time() - t0:.0f} s", flush=True)
            torch.save({"model": net.state_dict(), "base": args.base, "depth": args.depth, "inputs": 3, "iter": it}, args.out / "model.pt")
    net = net.cpu().eval()
    x = torch.zeros(1, 3, 64, 64)
    torch.onnx.export(net, x, args.out / "sharpen.onnx", input_names=["input"], output_names=["output"],
                      dynamic_axes={"input": {2: "h", 3: "w"}, "output": {2: "h", 3: "w"}}, opset_version=17, dynamo=False)
    meta = {"format": "crux-sharpen", "version": 1, "k": 3.0, "fwhm_scale": FWHM_SCALE, "multiple": 16,
            "inputs": ["asinh((v - sky) / (k * noise))", "fwhm / fwhm_scale", "sharpen (0 = correct only, 1 = also tighten)"],
            "truths": [t.name for t in truths], "iters": args.iters, "parameters": parameters(net),
            "max_scale": MAX_SCALE, "p_scale": P_SCALE, "flux_weight": FLUX_W}
    (args.out / "sharpen.json").write_text(json.dumps(finite(meta), indent=1, allow_nan=False), encoding="utf-8")
    print("exported", args.out / "sharpen.onnx", flush=True)


if __name__ == "__main__":
    main()
