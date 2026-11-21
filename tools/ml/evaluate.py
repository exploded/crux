"""Score a trained denoiser on bundles, against classical baselines.

    python tools/ml/evaluate.py --model .local/ml/runs/r1/model.pt \
        --bundles "C:/crux-work/train/2010-02-25 QHY9 NGC 3324"

Gain = how many times lower the true error is than the stack's own noise
(see train.score): ×2 is worth twice the frames. "faint" counts only pixels
near their patch's median, where noise shows most; "all" includes stars,
where blurring does harm.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import bundles as B
from model import UNet
from train import denoise, score

# Noise in each à trous (B3) layer for unit white noise (PixInsight's table).
LAYER_NOISE = [0.8907, 0.2007, 0.0856, 0.0413]


def wavelet(a: np.ndarray, sigma: float, k: tuple = (3.0, 2.0, 1.0, 0.5)) -> np.ndarray:
    """Multiscale (à trous B3) soft thresholding: the classical denoise."""
    x = torch.from_numpy(np.ascontiguousarray(a, np.float32))[None, None]
    b3 = torch.tensor([1, 4, 6, 4, 1], dtype=torch.float32) / 16
    out = torch.zeros_like(x)
    cur = x
    for j, kj in enumerate(k):
        d = 2**j
        pad = 2 * d
        s = F.conv2d(F.pad(cur, (pad, pad, 0, 0), mode="reflect"), b3.view(1, 1, 1, 5), dilation=(1, d))
        s = F.conv2d(F.pad(s, (0, 0, pad, pad), mode="reflect"), b3.view(1, 1, 5, 1), dilation=(d, 1))
        layer = cur - s
        t = kj * sigma * LAYER_NOISE[j]
        out += torch.sign(layer) * torch.clamp(layer.abs() - t, min=0)
        cur = s
    return (out + cur)[0, 0].numpy()


def star_flux(pairs: list[B.Pair], fn, min_snr: float = 30.0) -> dict:
    """Median ratio of denoised to original aperture flux (radius 2 FWHM,
    background from an annulus) for stars peaking ≥ min_snr σ, and of
    their peaks. 1.0 = kept."""
    flux, peak = [], []
    for p in pairs:
        r = max(3.0, 2.0 * p.fwhm)
        ro = int(np.ceil(r + 6))
        yy, xx = np.mgrid[-ro : ro + 1, -ro : ro + 1]
        d = np.hypot(yy, xx)
        inside, ring = d <= r, (d >= r + 2) & (d <= r + 6)
        for i in range(len(p.a)):
            a = p.a[i]
            dn = fn(p, a)
            h, w = a.shape
            core = a[ro:-ro, ro:-ro]
            # Local maxima in a 5×5 neighbourhood, bright enough.
            from numpy.lib.stride_tricks import sliding_window_view as sw
            win = sw(np.pad(a, 2, mode="edge"), (5, 5)).max((-1, -2))
            cand = np.argwhere((a == win) & (a - np.median(a) > min_snr * p.noise_a))
            for y, x in cand:
                if y < ro or x < ro or y >= h - ro or x >= w - ro:
                    continue
                ca, cd = a[y - ro : y + ro + 1, x - ro : x + ro + 1], dn[y - ro : y + ro + 1, x - ro : x + ro + 1]
                fa = (ca[inside] - np.median(ca[ring])).sum()
                fd = (cd[inside] - np.median(cd[ring])).sum()
                if fa > 0 and ca.max() < 0.95:
                    flux.append(fd / fa)
                    peak.append((cd[ro, ro] - np.median(cd[ring])) / (ca[ro, ro] - np.median(ca[ring])))
    if not flux:
        return {"stars": 0}
    return {"stars": len(flux), "flux": float(np.median(flux)), "flux_p10": float(np.percentile(flux, 10)), "flux_p90": float(np.percentile(flux, 90)), "peak": float(np.median(peak))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--bundles", nargs="+", type=Path, required=True)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.model, map_location=device)
    net = UNet(base=ck.get("base", 32), depth=ck.get("depth", 3)).to(device).eval()
    net.load_state_dict(ck["model"])
    pairs = B.load_all(args.bundles)
    groups = defaultdict(list)
    for p in pairs:
        groups[f"{p.camera} · {p.object} · {p.filter} · {p.kind}"].append(p)
    methods = {
        "network": lambda p, a: denoise(net, a, p.sky, p.noise_a, p.fwhm, device),
        "wavelet": lambda p, a: wavelet(a, p.noise_a),
        "gauss0.7": lambda p, a: B.blur(a, 0.7),
    }
    rows = {}
    print(f"{'stack':58s} " + "  ".join(f"{m:>17s}" for m in methods))
    print(f"{'':58s} " + "  ".join(f"{'all':>8s} {'faint':>8s}" for _ in methods))
    for g, ps in sorted(groups.items()) + [("ALL", pairs)]:
        r = {m: score(ps, fn) for m, fn in methods.items()}
        rows[g] = r
        print(f"{g[:58]:58s} " + "  ".join(f"{r[m]['all']:8.2f} {r[m]['faint']:8.2f}" for m in methods), flush=True)
    print("\nStar flux and peak after denoising, as a share of before (stars ≥ 30σ, not saturated):")
    for m in ("network", "wavelet"):
        f = star_flux(pairs, methods[m])
        rows[f"flux_{m}"] = f
        if f.get("stars"):
            print(f"  {m:8s} {f['stars']:5d} stars: flux {f['flux']:.4f} (10–90%: {f['flux_p10']:.3f}–{f['flux_p90']:.3f}), peak {f['peak']:.3f}")
    if args.json:
        args.json.write_text(json.dumps(rows, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
