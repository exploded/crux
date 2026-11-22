"""despike.py original.fits starless.fits out.fits [--n 60]
Prototype (2026-11-22, not used by crux yet): remove diffraction spike arms a star remover left behind.
1. Spike angles: the brightest stars' angular profiles (30-110 px) of the
   starless image, summed; the strongest direction and its +90/180/270.
2. Per star and arm: along the arm (0.5 px steps), the starless value on
   the line minus the median of two side strips 4-8 px either side, smoothed
   along the arm; subtracted with the arm's cross profile (Gaussian, width
   fitted once), only where positive.
"""
import argparse
import math
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
import bundles as B


def write_fits(path, img):
    h, w = img.shape
    cards = ["SIMPLE  =                    T", "BITPIX  =                  -32", "NAXIS   =                    2",
             f"NAXIS1  = {w:20d}", f"NAXIS2  = {h:20d}", "END"]
    hdr = "".join(c.ljust(80) for c in cards).encode("ascii")
    hdr += b" " * (-len(hdr) % 2880)
    data = img.astype(">f4").tobytes()
    data += b"\0" * (-len(data) % 2880)
    open(path, 'wb').write(hdr + data)


ap = argparse.ArgumentParser()
ap.add_argument('original')
ap.add_argument('starless')
ap.add_argument('out')
ap.add_argument('--n', type=int, default=60)
ap.add_argument('--length', type=int, default=160)
a = ap.parse_args()
o = B.read_fits_cube(a.original)[1][0].astype(np.float32)
s = B.read_fits_cube(a.starless)[1][0].astype(np.float32)
h, w = o.shape
sigma = B.noise_of(o[:2048, :2048])
mx = F.max_pool2d(torch.from_numpy(o)[None, None], 15, 1, 7)[0, 0].numpy()
ys, xs = np.nonzero((o == mx) & (o > np.median(o[::9, ::9]) + 200 * sigma))
order = np.argsort(-o[ys, xs])
stars = []
for i in order:
    y, x = int(ys[i]), int(xs[i])
    # Saturated cores are flat: many tied maxima per star.
    if all(abs(y - sy) > 20 or abs(x - sx) > 20 for sy, sx in stars):
        stars.append((y, x))
    if len(stars) >= a.n * 3:
        break
R = 110
# 1. Angles: summed angular profile over the brightest 20 stars (away from edges).
yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
rr = np.hypot(yy, xx)
ring = (rr >= 30) & (rr <= R)
ang = ((np.arctan2(yy, xx) + np.pi) / (2 * np.pi) * 720).astype(int) % 720  # 0.5 deg bins
tot = np.zeros(720)
used = 0
for y, x in stars:
    if y < R or x < R or y >= h - R or x >= w - R:
        continue
    c = s[y - R:y + R + 1, x - R:x + R + 1]
    p = np.bincount(ang[ring], c[ring].astype(np.float64), 720) / np.maximum(np.bincount(ang[ring], minlength=720), 1)
    p = p - np.median(p)
    tot += p / max(p.std(), 1e-12)
    used += 1
    if used >= 20:
        break
# Fold to 90 degrees (180 bins) and find the peak.
fold = tot.reshape(4, 180).sum(0)
k = int(np.argmax(np.convolve(np.r_[fold[-2:], fold, fold[:2]], np.ones(5) / 5, 'valid')))
theta0 = (k * 0.5) * math.pi / 180 - math.pi  # arctan2 convention offset
print(f"spike angle {math.degrees(theta0) % 90:.1f} deg (+90 multiples), from {used} stars; fold contrast {fold.max() / max(np.median(fold), 1e-9):.1f}")
out = s.copy()
SIG = 2.0  # arm width (Gaussian sigma, px)
# The line samples average +-1 px of the Gaussian: its mean there, not its peak.
LINE_GAIN = float(np.mean([math.exp(-0.5 * (d / SIG) ** 2) for d in (-1, -0.5, 0, 0.5, 1)]))
L = a.length
removed = 0.0
def arm_contrast(y, x, th):
    c, si = math.cos(th), math.sin(th)
    t = np.arange(15, 90, 1.0)
    X = np.round(x + t * c).astype(int); Y = np.round(y + t * si).astype(int)
    Xs = np.round(x + t * c - 12 * si).astype(int); Ys = np.round(y + t * si + 12 * c).astype(int)
    Xt = np.round(x + t * c + 12 * si).astype(int); Yt = np.round(y + t * si - 12 * c).astype(int)
    ok = (X >= 0) & (X < w) & (Y >= 0) & (Y < h) & (Xs >= 0) & (Xs < w) & (Ys >= 0) & (Ys < h) & (Xt >= 0) & (Xt < w) & (Yt >= 0) & (Yt < h)
    if ok.sum() < 20:
        return -1e9
    return float(np.median(s[Y[ok], X[ok]] - 0.5 * (s[Ys[ok], Xs[ok]] + s[Yt[ok], Xt[ok]])))


for y, x in stars[: a.n]:
    # This star's own angle within 4 degrees of the image's (field rotation,
    # the angle bins): the one where all four arms are brightest.
    best = max(np.arange(-4, 4.01, 0.25), key=lambda dd: sum(arm_contrast(y, x, theta0 + math.radians(dd) + q * math.pi / 2) for q in range(4)))
    for q in range(4):
        th = theta0 + math.radians(best) + q * math.pi / 2
        c, si = math.cos(th), math.sin(th)
        t = np.arange(12, L, 0.5)
        px, py = x + t * c, y + t * si
        nx, ny = -si, c  # across the arm
        def sample(img, off):
            X = np.round(px + off * nx).astype(int)
            Y = np.round(py + off * ny).astype(int)
            ok = (X >= 0) & (X < w) & (Y >= 0) & (Y < h)
            v = np.full(len(t), np.nan, np.float32)
            v[ok] = img[Y[ok], X[ok]]
            return v
        line = np.nanmean([sample(s, d) for d in (-1, -0.5, 0, 0.5, 1)], 0)
        side = np.nanmedian([sample(s, d) for d in (-16, -14, -12, -10, 10, 12, 14, 16)], 0)
        prof = line - side
        # Smooth along the arm (9 px) and keep the positive part.
        kern = np.ones(19) / 19
        prof = np.convolve(np.nan_to_num(prof), kern, 'same')
        prof = np.clip(prof, 0, None)
        # Only arms clearly above the noise.
        if np.nanmax(prof) < 1.0 * sigma:
            continue
        # Draw the arm once per pixel: its value at the pixel's distance
        # along the arm, times a Gaussian across it (sigma 1 px).
        x0, x1 = int(min(x, px.min())) - 5, int(max(x, px.max())) + 6
        y0, y1 = int(min(y, py.min())) - 5, int(max(y, py.max())) + 6
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
        if x1 <= x0 or y1 <= y0:
            continue
        YY, XX = np.mgrid[y0:y1, x0:x1]
        tt = (XX - x) * c + (YY - y) * si
        dd = -(XX - x) * si + (YY - y) * c
        m = (np.abs(dd) <= 7) & (tt >= t[0]) & (tt <= t[-1])
        model = np.zeros(XX.shape, np.float32)
        model[m] = np.interp(tt[m], t, prof) * np.exp(-0.5 * (dd[m] / SIG) ** 2) / LINE_GAIN
        out[y0:y1, x0:x1] -= model
        removed += float(prof.sum())
print(f"removed {removed / sigma:.0f} sigma-px along arms")
write_fits(a.out, out)
print('wrote', a.out)
