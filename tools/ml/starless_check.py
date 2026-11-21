"""Check a starless image against the original and crux's classic starless.

- Away from stars (pixels the classic fill left alone), the network should
  change nothing: mean and spread of (network − original) there, in noise σ.
- Where stars were, the light removed should be about what the classic fill
  removed (the classic fill is a rough reference, not a truth).

    python starless_check.py original.fits classic.fits network.fits [--png out.png --at X Y]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import bundles as B
from starcheck import mtf, png


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("original", type=Path)
    ap.add_argument("classic", type=Path)
    ap.add_argument("network", type=Path)
    ap.add_argument("--png", type=Path)
    ap.add_argument("--at", type=int, nargs=2, default=None, help="crop top-left x y")
    ap.add_argument("--size", type=int, default=400)
    args = ap.parse_args()
    o, c, n = (B.read_fits_cube(p)[1][0] for p in (args.original, args.classic, args.network))
    h, w = o.shape
    y0, x0 = h // 20, w // 20
    core = (slice(y0, h - y0), slice(x0, w - x0))
    o, c, n = o[core], c[core], n[core]
    sigma = B.noise_of(o[: min(2048, o.shape[0]), : min(2048, o.shape[1])])
    untouched = np.abs(c - o) < 1e-7
    stars = (o - c) > 3 * sigma
    d = (n - o)[untouched] / sigma
    print(f"noise σ {sigma:.3e}; untouched pixels {untouched.mean() * 100:.1f}%, star pixels {stars.mean() * 100:.1f}%")
    print(f"  away from stars: network − original mean {d.mean():+.3f}σ, spread {d.std():.3f}σ")
    rem_c, rem_n = (o - c)[stars].sum(), (o - n)[stars].sum()
    print(f"  on stars: network removes {rem_n / rem_c * 100:.0f}% of the light the classic fill removes")
    resid = (n - c)[stars] / sigma
    print(f"  on stars: network − classic median {np.median(resid):+.2f}σ (positive = brighter than the fill)")
    if args.png:
        s = args.size
        if args.at:
            xa, ya = args.at[0] - x0, args.at[1] - y0
        else:
            ya, xa = (o.shape[0] - s) // 2, (o.shape[1] - s) // 2
        crops = [v[ya : ya + s, xa : xa + s] for v in (o, c, n)]
        a = crops[0]
        med = float(np.median(a))
        sig = 1.4826 * float(np.median(np.abs(a - med)))
        c0 = max(0.0, med - 2.8 * sig)
        m = float(mtf(np.array((med - c0) / (1 - c0)), 0.25))
        st = lambda v: (np.clip(mtf(np.clip((v - c0) / (1 - c0), 0, 1), m), 0, 1) * 255).round().astype(np.uint8)
        gap = 8
        out = np.full((s, 3 * s + 2 * gap), 40, np.uint8)
        for i, v in enumerate(crops):
            out[:, i * (s + gap) : i * (s + gap) + s] = st(v)
        png(args.png, out)
        print("wrote", args.png)


if __name__ == "__main__":
    main()
