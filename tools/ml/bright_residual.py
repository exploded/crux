"""Light a star remover leaves at the brightest stars.

For the `--n` brightest stars of the original (local maxima, kept apart),
the starless image's mean within `--inner` px (centre excluded inside 3 px)
less the median of the ring from `--inner` to `--outer` px, in noise σ of
the original: positive = a glow left behind, negative = a hole. Uses crux's
own outputs only.

    python bright_residual.py original.fits starless.fits [starless2.fits ...] [--n 20]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import bundles as B


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("original", type=Path)
    ap.add_argument("starless", type=Path, nargs="+")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--inner", type=int, default=40)
    ap.add_argument("--outer", type=int, default=70)
    args = ap.parse_args()
    o = B.read_fits_cube(args.original)[1][0].astype(np.float32)
    h, w = o.shape
    sigma = B.noise_of(o[: min(2048, h), : min(2048, w)])
    R, r_in = args.outer, args.inner
    mx = F.max_pool2d(torch.from_numpy(o)[None, None], 15, 1, 7)[0, 0].numpy()
    ys, xs = np.nonzero((o == mx) & (o > np.median(o[::9, ::9]) + 200 * sigma))
    order = np.argsort(-o[ys, xs])
    centres: list[tuple[int, int]] = []
    for i in order:
        y, x = int(ys[i]), int(xs[i])
        if y < R or x < R or y >= h - R or x >= w - R:
            continue
        if any(abs(y - cy) < 2 * R and abs(x - cx) < 2 * R for cy, cx in centres):
            continue
        centres.append((y, x))
        if len(centres) >= args.n:
            break
    yy, xx = np.mgrid[-R : R + 1, -R : R + 1]
    rr = np.sqrt(yy * yy + xx * xx)
    inner, ring = (rr >= 3) & (rr < r_in), (rr >= r_in) & (rr <= R)
    print(f"{len(centres)} brightest stars, noise σ {sigma:.3e}")
    for p in args.starless:
        s = B.read_fits_cube(p)[1][0].astype(np.float32)
        v = []
        for y, x in centres:
            c = s[y - R : y + R + 1, x - R : x + R + 1]
            v.append((float(c[inner].mean()) - float(np.median(c[ring]))) / sigma)
        v = np.array(v)
        print(f"  {p.stem[:40]:40s} glow median {np.median(v):+.2f}σ, mean {v.mean():+.2f}σ, worst {v.max():+.2f}σ, deepest hole {v.min():+.2f}σ")


if __name__ == "__main__":
    main()
