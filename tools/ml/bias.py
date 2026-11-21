"""Signal kept by the denoiser, by brightness.

A is an unbiased view of the scene, so mean(D − A) over a set of pixels is
the denoiser's bias there, as long as the set isn't chosen from A's own
noise: the bands come from B, whose noise is independent. Reported as a
share of the signal above the patch's median. mean(A − B) is shown too: it
is how well B was matched to A (zero if perfect), not the denoiser.

    python tools/ml/bias.py --model <run>/model.pt --bundles <bundle>...
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

import bundles as B
from model import UNet
from train import denoise

BANDS = [(3, 10), (10, 30), (30, 100), (100, 300), (300, 1e9)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--bundles", nargs="+", type=Path, required=True)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.model, map_location=device)
    net = UNet(base=ck.get("base", 32), depth=ck.get("depth", 3)).to(device).eval()
    net.load_state_dict(ck["model"])
    acc = {b: [0.0, 0.0, 0, 0.0] for b in BANDS}  # sum(D−A), sum(B−median), pixels, sum(A−B)
    for p in B.load_all(args.bundles):
        for i in range(len(p.a)):
            a, b = p.a[i], p.b[i]
            d = denoise(net, a, p.sky, p.noise_a, p.fwhm, device)
            med = np.median(b)
            # Bands from a lightly smoothed B, so noise doesn't sort pixels.
            snr = (B.blur(b, 1.5) - med) / p.noise_b
            for lo, hi in BANDS:
                m = (snr >= lo) & (snr < hi)
                if m.any():
                    acc[(lo, hi)][0] += float((d - a)[m].sum())
                    acc[(lo, hi)][1] += float((b - med)[m].sum())
                    acc[(lo, hi)][2] += int(m.sum())
                    acc[(lo, hi)][3] += float((a - b)[m].sum())
    print(f"{'signal (σ)':>14s} {'pixels':>10s} {'bias (D−A)':>11s} {'match (A−B)':>12s}")
    for (lo, hi), (da, sig, n, ab) in acc.items():
        if n:
            print(f"{lo:>6.0f}–{hi if hi < 1e8 else float('inf'):<7.0f} {n:10d} {da / sig * 100:+10.2f}% {ab / sig * 100:+11.2f}%")


if __name__ == "__main__":
    main()
