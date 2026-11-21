"""Compare stars before and after a crux network, from FITS exports:
total flux in a wide aperture (sharpening moves light, it mustn't make or
lose it), and a corner crop for looking at.

    python starcheck.py original.fits processed.fits [more.fits] --png out.png
"""

from __future__ import annotations

import argparse
import struct
import zlib
from pathlib import Path

import numpy as np

import bundles as B


def png(path: Path, img: np.ndarray):
    h, w = img.shape
    raw = b"".join(b"\0" + img[y].tobytes() for y in range(h))
    chunk = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def mtf(x, m):
    return np.where(x <= 0, 0, np.where(x >= 1, 1, (m - 1) * x / ((2 * m - 1) * x - m)))


def stars(img: np.ndarray, sky: float, sigma: float, n: int = 600) -> np.ndarray:
    from numpy.lib.stride_tricks import sliding_window_view as sw

    win = sw(np.pad(img, 4, mode="edge"), (9, 9)).max((-1, -2))
    pk = np.argwhere((img == win) & (img - sky > 40 * sigma) & (img < 0.8))
    vals = img[pk[:, 0], pk[:, 1]]
    return pk[np.argsort(-vals)[:n]]


def flux(img, y, x, r=16, r_in=22, r_out=30):
    yy, xx = np.mgrid[-r_out : r_out + 1, -r_out : r_out + 1]
    d = np.hypot(yy, xx)
    c = img[y - r_out : y + r_out + 1, x - r_out : x + r_out + 1]
    if c.shape != d.shape or (c <= 0).any():
        return np.nan
    bg = np.median(c[(d >= r_in) & (d <= r_out)])
    return float((c[d <= r] - bg).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--png", type=Path)
    ap.add_argument("--corner", type=int, nargs=2, default=[250, 170], help="crop top-left x y")
    ap.add_argument("--size", type=int, default=360)
    ap.add_argument("--r", type=int, default=16, help="aperture radius (ring from r+6 to r+14)")
    args = ap.parse_args()
    imgs = [B.read_fits_cube(f)[1][0] for f in args.files]
    a = imgs[0]
    sky = float(np.median(a[::9, ::9]))
    sigma = B.noise_of(a[1000:2000, 1000:2000])
    pk = stars(a, sky, sigma)
    R = args.r
    base = np.array([flux(a, y, x, R, R + 6, R + 14) for y, x in pk])
    print(f"{len(pk)} stars ≥ 40σ, not saturated")
    for f, im in zip(args.files[1:], imgs[1:]):
        r = np.array([flux(im, y, x, R, R + 6, R + 14) for y, x in pk]) / base
        r = r[np.isfinite(r)]
        print(f"  {f.name}: flux kept median {np.median(r):.4f} (10–90%: {np.percentile(r, 10):.3f}–{np.percentile(r, 90):.3f})")
    if args.png:
        x0, y0, n = args.corner[0], args.corner[1], args.size
        crops = [im[y0 : y0 + n, x0 : x0 + n] for im in imgs]
        c = crops[0]
        med = float(np.median(c))
        sig = 1.4826 * float(np.median(np.abs(c - med)))
        c0 = max(0.0, med - 2.8 * sig)
        m = float(mtf(np.array((med - c0) / (1 - c0)), 0.25))
        st = lambda v: np.kron((np.clip(mtf(np.clip((v - c0) / (1 - c0), 0, 1), m), 0, 1) * 255).round().astype(np.uint8), np.ones((2, 2), np.uint8))
        tiles = [st(v) for v in crops]
        gap = 10
        out = np.full((tiles[0].shape[0], sum(t.shape[1] for t in tiles) + gap * (len(tiles) - 1)), 40, np.uint8)
        x = 0
        for t in tiles:
            out[:, x : x + t.shape[1]] = t
            x += t.shape[1] + gap
        png(args.png, out)
        print("wrote", args.png)


if __name__ == "__main__":
    main()
