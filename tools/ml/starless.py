"""Train crux's star remover.

Backgrounds are crux's classic starless (stars::starless, `crux dev
starless`) of crux's own masters, denoised with crux's denoiser. Each
training input is a background patch with real stars painted on (star
cutouts from the bundles' PSF samples, random places and brightnesses) plus
real stack noise; the target is the same background with the same noise.
So the network removes stars and leaves the noise alone, as crux's natural
star colours need.

Inputs: [asinh image, FWHM / 4]. Never uses any other program's output.

With --real-bright, a share of patches also get real bright stars: cut
from crux's denoised masters (the star corrector's cache) where they sit
on flat sky, with their halos, diffraction spikes and saturated cores, the
sky under them fitted on a ring and removed. Synthetic halos and spikes
(--bright, run s2) looked like nebula and taught the network to touch it;
real ones come with the core that explains them.

    python starless.py --backgrounds C:/crux-work/train/starless --bundles C:/crux-work/train \
        --denoiser ../../.local/ml/runs/r2-holdout-stl-qhy9/model.pt --out ../../.local/ml/runs/s1
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
from deconv import crux_fwhm, denoise_full, noise_bank
from model import FWHM_SCALE, UNet, parameters
from train import finite

CROP = 128


class Background:
    def __init__(self, name, img, sky, sigma, fwhm):
        self.name, self.img, self.sky, self.sigma, self.fwhm = name, img, sky, sigma, fwhm


def load_backgrounds(folder: Path, denoiser: Path, cache: Path, device, workspaces: list[Path]) -> list[Background]:
    cache.mkdir(parents=True, exist_ok=True)
    ck = torch.load(denoiser, map_location=device)
    net = UNet(base=ck.get("base", 32), depth=ck.get("depth", 3)).to(device).eval()
    net.load_state_dict(ck["model"])
    known = crux_fwhm(workspaces)
    out = []
    for f in sorted(Path(folder).glob("*.fits")):
        stem = f.stem.split("_", 1)[1] if not f.stem.startswith("masterLight") else f.stem
        c = cache / (f.stem + ".npz")
        if c.is_file() and c.stat().st_mtime > f.stat().st_mtime:
            z = np.load(c)
            out.append(Background(f.stem, z["img"], float(z["sky"]), float(z["sigma"]), known.get(stem, 5.0)))
            continue
        _, a = B.read_fits_cube(f)
        a = a[0]
        h, w = a.shape
        y0, x0 = h // 20, w // 20  # registration edges
        a = np.ascontiguousarray(a[y0 : h - y0, x0 : w - x0])
        sky = float(np.median(a[::7, ::7]))
        sigma = B.noise_of(a[:2048, :2048])
        fw = known.get(stem, 5.0)
        d = denoise_full(net, a, sky, sigma, fw, device)
        np.savez(c, img=d.astype(np.float32), sky=sky, sigma=sigma)
        print(f"  {f.stem}: {a.shape[1]}×{a.shape[0]}, noise {sigma:.2e}, FWHM {fw:.2f}", flush=True)
        out.append(Background(f.stem, d, sky, sigma, fw))
    return out


def psf_bank(roots: list[Path], exclude: list[str]) -> list[tuple[np.ndarray, float]]:
    """Star stamps (background removed, sum 1) with their FWHM, from the
    bundles' PSF samples."""
    out = []
    for r in roots:
        for man in sorted(Path(r).rglob("manifest.json")):
            if any(e.lower() in man.parent.name.lower() for e in exclude):
                continue
            m = json.loads(man.read_text(encoding="utf-8"))
            for s in m["sets"]:
                psf = s.get("psf")
                if not psf:
                    continue
                _, cube = B.read_fits_cube(man.parent / psf["file"])
                for st, cut in zip(psf["stars"], cube):
                    c = np.clip(cut - st["background"], 0, None)
                    tot = c.sum()
                    if tot <= 0 or not math.isfinite(tot):
                        continue
                    out.append(((c / tot).astype(np.float32), float(st["fwhm"])))
    return out


def real_bright_bank(cache: Path, exclude: list[str], radius: int = 64, min_peak: float = 300.0, per_master: int = 40, floor: float = 1.5) -> list[np.ndarray]:
    """Real bright stars (sky removed, edges tapered to 0) from denoised
    masters (`deconv.py`'s cache: img, sky, sigma). A star qualifies when its
    peak is ≥ `min_peak` noise σ above the sky and the ring at 0.85–1 of
    `radius` is flat: its eight sectors' medians agree within 1.5σ after a
    plane is removed, so no nebula comes along. Light under `floor` σ of the
    original noise is cut away (soft threshold)."""
    out = []
    r = radius
    yy, xx = np.mgrid[-r : r + 1, -r : r + 1].astype(np.float32)
    rr = np.sqrt(yy * yy + xx * xx)
    ring = (rr >= 0.85 * r) & (rr <= r)
    sector = ((np.arctan2(yy, xx) + math.pi) / (2 * math.pi) * 8).astype(int) % 8
    taper = np.clip((r - rr) / (0.25 * r), 0, 1)
    taper = (0.5 - 0.5 * np.cos(math.pi * taper)).astype(np.float32)
    A = np.stack([np.ones(ring.sum()), xx[ring], yy[ring]], 1)
    for f in sorted(Path(cache).glob("*.npz")):
        if any(e.lower() in f.stem.lower() for e in exclude):
            continue
        z = np.load(f)
        img, sky, sigma = z["img"].astype(np.float32), float(z["sky"]), float(z["sigma"])
        h, w = img.shape
        # Local maxima (7 × 7) far above the sky, brightest first.
        t = torch.from_numpy(img)[None, None]
        mx = F.max_pool2d(t, 7, 1, 3)[0, 0].numpy()
        ys, xs = np.nonzero((img == mx) & (img > sky + min_peak * sigma))
        order = np.argsort(-img[ys, xs])
        taken: list[tuple[int, int]] = []
        n = 0
        for i in order:
            y, x = int(ys[i]), int(xs[i])
            if y < r or x < r or y >= h - r or x >= w - r:
                continue
            if any(abs(y - ty) <= r and abs(x - tx) <= r for ty, tx in taken):
                continue
            cut = img[y - r : y + r + 1, x - r : x + r + 1]
            v = cut[ring].astype(np.float64)
            keep = np.ones(len(v), bool)
            for _ in range(3):
                coef, *_ = np.linalg.lstsq(A[keep], v[keep], rcond=None)
                res = v - A @ coef
                s_ = 1.4826 * np.median(np.abs(res[keep] - np.median(res[keep]))) + 1e-12
                keep = np.abs(res) < 3 * s_
            med = [np.median(res[(sector[ring] == k) & keep]) for k in range(8) if ((sector[ring] == k) & keep).any()]
            if len(med) < 8 or max(med) - min(med) > 1.5 * sigma or math.hypot(coef[1], coef[2]) * 2 * r > 3 * sigma:
                continue
            plane = (coef[0] + coef[1] * xx + coef[2] * yy).astype(np.float32)
            if cut[r, r] - plane[r, r] < min_peak * sigma:
                continue
            # Faint structure (denoised sky, nebula residue) under 1.5σ of
            # the original noise goes: real halos and spikes are far above it.
            out.append((np.maximum(cut - plane - floor * sigma, 0) * taper).astype(np.float32))
            taken.append((y, x))
            n += 1
            if n >= per_master:
                break
        print(f"  {f.stem}: {n} bright stars", flush=True)
    return out


MAX_PEAK = 3000.0  # brightest star peak, noise σ
BRIGHT = False  # add halos and diffraction spikes to bright stars
P_BRIGHT = 1.0  # share of patches that get them
REAL: list[np.ndarray] = []  # real bright stars (`real_bright_bank`)
P_REAL = 0.0  # share of patches that get one to three of them


def bright(stars: np.ndarray, y0: int, x0: int, peak: float, fw: float, angle: float, rng: random.Random):
    """A bright star's extended light: a Moffat halo (most of them) and four
    diffraction spikes (some), drawn into `stars` around (y0, x0)."""
    h, w = stars.shape
    R = int(min(60, 6 * fw + 20 * math.log10(peak / 1e-4 + 1) / 4))
    ya, yb, xa, xb = max(0, y0 - R), min(h, y0 + R + 1), max(0, x0 - R), min(w, x0 + R + 1)
    if yb <= ya or xb <= xa:
        return
    yy, xx = np.mgrid[ya:yb, xa:xb].astype(np.float32)
    dy, dx = yy - y0, xx - x0
    if rng.random() < 0.7:
        alpha = fw * rng.uniform(1.0, 2.5)
        halo = (1 + (dy * dy + dx * dx) / alpha**2) ** -2.5
        stars[ya:yb, xa:xb] += peak * rng.uniform(0.005, 0.05) * halo
    if rng.random() < 0.4:
        L = fw * rng.uniform(4, 14) * max(1.0, math.log10(peak / 1e-4))
        a0 = peak * rng.uniform(0.003, 0.03)
        for k in range(4):
            th = angle + k * math.pi / 2
            c, s = math.cos(th), math.sin(th)
            t = dx * c + dy * s  # along the spike
            u = -dx * s + dy * c  # across it
            ray = np.where(t > 0, np.exp(-t / max(L / 3, 1.0)), 0.0) * np.exp(-0.5 * (u / 0.7) ** 2)
            stars[ya:yb, xa:xb] += a0 * ray.astype(np.float32)


def batch(bgs, psfs, bank, n, rng, device):
    xs, ys = [], []
    for _ in range(n):
        g = rng.choice(bgs)
        f = rng.uniform(1.0, 2.2) if rng.random() < 0.6 else 1.0
        c = int(math.ceil(CROP * f)) + 1
        h, w = g.img.shape
        y, x = rng.randrange(h - c), rng.randrange(w - c)
        bg = g.img[y : y + c, x : x + c].astype(np.float32).copy()
        sigma0 = g.sigma
        # Stars: brightness log-uniform from 5σ to 3000σ peak, count varied.
        stars = np.zeros_like(bg)
        spike_angle = rng.uniform(0, math.pi / 2)
        # Bright stars (halos, spikes, up to MAX_PEAK) in a share of patches;
        # the rest keep the plain mix (peaks up to 3000σ).
        bright_patch = BRIGHT and rng.random() < P_BRIGHT
        top = MAX_PEAK if bright_patch else min(MAX_PEAK, 3000.0)
        count = int(rng.choice([rng.uniform(2, 15), rng.uniform(15, 60), rng.uniform(60, 160)]) * (c / 128) ** 2)
        fws = []
        for _ in range(count):
            stamp, fw = rng.choice(psfs)
            k = rng.randrange(4)
            stamp = np.rot90(stamp, k)
            sh = stamp.shape[0]
            peak_snr = math.exp(rng.uniform(math.log(5), math.log(top)))
            amp = peak_snr * sigma0 / max(float(stamp.max()), 1e-9)
            cy, cx = rng.randrange(-sh // 2, c - sh // 2), rng.randrange(-sh // 2, c - sh // 2)
            ya, xa = max(0, cy), max(0, cx)
            yb, xb = min(c, cy + sh), min(c, cx + sh)
            if yb <= ya or xb <= xa:
                continue
            stars[ya:yb, xa:xb] += amp * stamp[ya - cy : yb - cy, xa - cx : xb - cx]
            fws.append(fw)
            if bright_patch and peak_snr > 300:
                bright(stars, cy + sh // 2, cx + sh // 2, peak_snr * sigma0, fw, spike_angle, rng)
        if REAL and rng.random() < P_REAL:
            for _ in range(rng.choice([1, 1, 2, 3])):
                st = np.rot90(rng.choice(REAL), rng.randrange(4))
                if rng.random() < 0.5:
                    st = st[:, ::-1]
                sh = st.shape[0]
                cy, cx = rng.randrange(-sh // 3, c - 2 * sh // 3), rng.randrange(-sh // 3, c - 2 * sh // 3)
                ya, xa = max(0, cy), max(0, cx)
                yb, xb = min(c, cy + sh), min(c, cx + sh)
                if yb > ya and xb > xa:
                    stars[ya:yb, xa:xb] += rng.uniform(0.3, 1.2) * st[ya - cy : yb - cy, xa - cx : xb - cx]
        img = np.minimum(bg + stars, 1.0)
        t = torch.from_numpy(np.stack([img, bg]))[None].to(device)
        if f > 1.0:
            t = F.adaptive_avg_pool2d(t, (CROP, CROP))
        else:
            t = t[..., :CROP, :CROP]
        if rng.random() < 0.5:
            t = t.flip(-1)
        t = torch.rot90(t, rng.randrange(4), (-2, -1))
        sigma = sigma0 * rng.uniform(0.3, 1.5) / f
        nz = rng.choice(bank)
        ny, nx = rng.randrange(nz.shape[0] - CROP + 1), rng.randrange(nz.shape[1] - CROP + 1)
        noise = torch.from_numpy(np.rot90(nz[ny : ny + CROP, nx : nx + CROP], rng.randrange(4)).copy()).to(device) * sigma
        kk = 3.0 * sigma
        xi = torch.asinh((t[0, 0] + noise - g.sky) / kk)[None, None]
        yt = torch.asinh((t[0, 1] + noise - g.sky) / kk)[None, None]
        fw_in = (float(np.median(fws)) if fws else g.fwhm) / f
        xs.append(torch.cat([xi, torch.full_like(xi, fw_in / FWHM_SCALE)], 1))
        ys.append(yt)
    return torch.cat(xs), torch.cat(ys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backgrounds", type=Path, required=True)
    ap.add_argument("--bundles", type=Path, nargs="+", required=True)
    ap.add_argument("--exclude", nargs="*", default=["NGC 6188", "QHY9"])
    ap.add_argument("--denoiser", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--iters", type=int, default=15000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-peak", type=float, default=3000.0)
    ap.add_argument("--bright", action="store_true", help="halos and diffraction spikes on bright stars")
    ap.add_argument("--p-bright", type=float, default=1.0, help="share of patches with bright stars")
    ap.add_argument("--init", type=Path, help="start from this checkpoint (fine-tune)")
    ap.add_argument("--workspaces", type=Path, nargs="+", default=[Path("C:/crux-work"), Path("D:/crux-work")], help="where crux-integration.json gives each master's FWHM")
    ap.add_argument("--real-bright", type=Path, help="denoised masters (deconv.py's cache) to cut real bright stars from")
    ap.add_argument("--p-real", type=float, default=0.4, help="share of patches with real bright stars")
    ap.add_argument("--real-radius", type=int, default=64, help="real bright star cutout radius, px")
    ap.add_argument("--real-floor", type=float, default=1.5, help="light under this many noise σ is cut from real bright stars")
    args = ap.parse_args()
    global MAX_PEAK, BRIGHT, P_BRIGHT, REAL, P_REAL
    MAX_PEAK, BRIGHT, P_BRIGHT = args.max_peak, args.bright, args.p_bright
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    print("backgrounds:", flush=True)
    bgs = load_backgrounds(args.backgrounds, args.denoiser, args.out.parent / "starless_cache", device, args.workspaces)
    if args.real_bright:
        print("real bright stars:", flush=True)
        REAL, P_REAL = real_bright_bank(args.real_bright, args.exclude, radius=args.real_radius, floor=args.real_floor), args.p_real
    psfs = psf_bank(args.bundles, args.exclude)
    bank = noise_bank(args.bundles, args.exclude)
    print(f"{len(bgs)} backgrounds, {len(psfs)} star stamps, {len(REAL)} real bright stars, {len(bank)} noise patches", flush=True)
    vrng = random.Random(999)
    val = [batch(bgs, psfs, bank, args.batch, vrng, device) for _ in range(8)]
    net = UNet(base=args.base, depth=args.depth, inputs=2).to(device)
    if args.init:
        net.load_state_dict(torch.load(args.init, map_location=device)["model"])
        print(f"starting from {args.init}", flush=True)
    print(f"U-Net depth {args.depth}, {parameters(net) / 1e6:.2f} M parameters", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.iters, pct_start=0.05)
    args.out.mkdir(parents=True, exist_ok=True)
    t0, run = time.time(), 0.0
    for it in range(1, args.iters + 1):
        x, y = batch(bgs, psfs, bank, args.batch, rng, device)
        loss = F.l1_loss(net(x), y)
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
            torch.save({"model": net.state_dict(), "base": args.base, "depth": args.depth, "inputs": 2, "iter": it}, args.out / "model.pt")
    net = net.cpu().eval()
    x = torch.zeros(1, 2, 64, 64)
    torch.onnx.export(net, x, args.out / "starless.onnx", input_names=["input"], output_names=["output"],
                      dynamic_axes={"input": {2: "h", 3: "w"}, "output": {2: "h", 3: "w"}}, opset_version=17, dynamo=False)
    meta = {"format": "crux-starless", "version": 1, "k": 3.0, "fwhm_scale": FWHM_SCALE, "multiple": 2 ** args.depth * 2,
            "inputs": ["asinh((v - sky) / (k * noise))", "fwhm / fwhm_scale"],
            "backgrounds": [g.name for g in bgs], "iters": args.iters, "parameters": parameters(net),
            "max_peak": MAX_PEAK, "bright": BRIGHT, "real_bright": len(REAL), "p_real": P_REAL, "real_radius": args.real_radius, "real_floor": args.real_floor}
    (args.out / "starless.json").write_text(json.dumps(finite(meta), indent=1, allow_nan=False), encoding="utf-8")
    print("exported", args.out / "starless.onnx", flush=True)


if __name__ == "__main__":
    main()
