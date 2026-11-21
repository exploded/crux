"""Read crux training bundles (`crux collect`) and serve training pairs.

A bundle is a folder with manifest.json and Float32 FITS cubes, one plane
per patch. Pixel values are as crux stores them (ADU / adu_scale).
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FORMAT = "crux-training-bundle"
SUPPORTED = {1}


def read_fits_cube(path: Path) -> tuple[dict, np.ndarray]:
    """A primary-HDU FITS file as (header, array shaped [planes, h, w])."""
    data = Path(path).read_bytes()
    hdr: dict[str, object] = {}
    pos = 0
    done = False
    while not done:
        block = data[pos : pos + 2880]
        pos += 2880
        for k in range(36):
            card = block[k * 80 : (k + 1) * 80].decode("ascii", "replace")
            key = card[:8].strip()
            if key == "END":
                done = True
                break
            if card[8:10] != "= ":
                continue
            raw = card[10:].split("/")[0].strip() if not card[10:].lstrip().startswith("'") else card[10:]
            if raw.lstrip().startswith("'"):
                s = raw.lstrip()[1:]
                hdr[key] = s[: s.find("'")].rstrip() if "'" in s else s
            elif raw in ("T", "F"):
                hdr[key] = raw == "T"
            else:
                try:
                    hdr[key] = int(raw)
                except ValueError:
                    try:
                        hdr[key] = float(raw.replace("D", "E"))
                    except ValueError:
                        hdr[key] = raw
    bitpix = int(hdr["BITPIX"])
    if bitpix != -32:
        raise ValueError(f"{path}: BITPIX {bitpix}, expected -32")
    w, h = int(hdr["NAXIS1"]), int(hdr["NAXIS2"])
    n = int(hdr.get("NAXIS3", 1)) if int(hdr["NAXIS"]) == 3 else 1
    arr = np.frombuffer(data, dtype=">f4", count=w * h * n, offset=pos).astype(np.float32)
    return hdr, arr.reshape(n, h, w)


@dataclass
class Pair:
    """One pair of stacks' patches: a[i] and b[i] are the same place."""

    bundle: str
    set_id: str
    camera: str
    object: str
    filter: str
    kind: str
    a: np.ndarray  # [n, p, p]
    b: np.ndarray
    noise_a: float
    noise_b: float
    sky: float
    fwhm: float  # pixels, of the stack


def load_bundle(folder: Path) -> list[Pair]:
    folder = Path(folder)
    m = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if m.get("format") != FORMAT or m.get("version") not in SUPPORTED:
        raise ValueError(f"{folder}: not a supported bundle ({m.get('format')} v{m.get('version')})")
    pairs = []
    for s in m["sets"]:
        psf = s.get("psf") or {}
        fw = [st["fwhm"] for st in psf.get("stars", []) if st.get("fwhm")]
        for p in s["pairs"]:
            _, a = read_fits_cube(folder / p["file_a"])
            _, b = read_fits_cube(folder / p["file_b"])
            fwhm = float(np.median(fw)) if fw else float(p["a"].get("fwhm") or 3.0)
            pairs.append(
                Pair(
                    bundle=folder.name,
                    set_id=s["id"],
                    camera=s.get("camera") or "unknown",
                    object=s["object"],
                    filter=s["filter"],
                    kind=p["kind"],
                    a=a,
                    b=b,
                    noise_a=float(p["a"]["noise"]),
                    noise_b=float(p["b"]["noise"]),
                    sky=float(p["sky"]),
                    fwhm=fwhm,
                )
            )
    return pairs


def load_all(roots: list[Path]) -> list[Pair]:
    """Every bundle under the given folders (a folder with manifest.json)."""
    out = []
    for r in roots:
        r = Path(r)
        found = [r] if (r / "manifest.json").is_file() else sorted(p.parent for p in r.rglob("manifest.json"))
        for f in found:
            out.extend(load_bundle(f))
    return out


# ------------------------------------------------------------- transforms

K = 3.0  # asinh knee, in noise σ: linear within about ±K σ of the sky.


def forward(v: np.ndarray, sky: float, sigma: float) -> np.ndarray:
    """Stored values → the network's domain: noise ≈ 1 near the sky,
    bright stars compressed."""
    return np.arcsinh((v - sky) / (K * sigma))


def inverse(y: np.ndarray, sky: float, sigma: float) -> np.ndarray:
    return sky + K * sigma * np.sinh(y)


_B3 = np.array([1, 4, 6, 4, 1], np.float32) / 16


def noise_of(img: np.ndarray) -> float:
    """crux's noise estimator, robust version: finest à trous layer,
    1.4826 MAD / 0.8907."""
    from numpy.lib.stride_tricks import sliding_window_view as sw

    if min(img.shape) < 8:
        return float("nan")
    s = (sw(img, 5, axis=1) * _B3).sum(-1)
    s = (sw(s, 5, axis=0) * _B3).sum(-1)
    layer = img[2:-2, 2:-2] - s
    med = np.median(layer)
    return float(1.4826 * np.median(np.abs(layer - med)) / 0.8907)


# ------------------------------------------------------------ augmentation


def _gauss_kernel(sigma: float) -> np.ndarray:
    r = max(1, int(math.ceil(3 * sigma)))
    x = np.arange(-r, r + 1, dtype=np.float32)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def blur(img: np.ndarray, sigma: float) -> np.ndarray:
    from numpy.lib.stride_tricks import sliding_window_view as sw

    k = _gauss_kernel(sigma)
    r = len(k) // 2
    p = np.pad(img, r, mode="reflect")
    t = (sw(p, len(k), axis=1) * k).sum(-1)
    return (sw(t, len(k), axis=0) * k).sum(-1)


def downscale(img: np.ndarray, f: float) -> np.ndarray:
    """Area-average down by any factor ≥ 1 (like binning, for any plate
    scale): each output pixel is the mean of the input area it covers."""
    import torch
    import torch.nn.functional as F

    h, w = img.shape
    t = torch.from_numpy(np.ascontiguousarray(img, np.float32))[None, None]
    return F.adaptive_avg_pool2d(t, (max(1, round(h / f)), max(1, round(w / f))))[0, 0].numpy()


def bin2(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[0] // 2 * 2, img.shape[1] // 2 * 2
    return img[:h, :w].reshape(h // 2, 2, w // 2, 2).mean((1, 3))


def bayer(img: np.ndarray, colour: str) -> np.ndarray:
    """Keep one colour's Bayer sites (RGGB) and fill the rest bilinearly, as
    a debayer does: the correlated noise of one-shot colour cameras."""
    h, w = img.shape
    mask = np.zeros((h, w), bool)
    if colour == "r":
        mask[0::2, 0::2] = True
    elif colour == "b":
        mask[1::2, 1::2] = True
    else:
        mask[0::2, 1::2] = True
        mask[1::2, 0::2] = True
    if colour == "g":
        k = np.array([[0, 1, 0], [1, 4, 1], [0, 1, 0]], np.float32) / 4
    else:
        k = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], np.float32) / 4
    from numpy.lib.stride_tricks import sliding_window_view as sw

    v = np.where(mask, img, 0).astype(np.float32)
    out = (sw(np.pad(v, 1, mode="reflect"), (3, 3)) * k).sum((-1, -2))
    return np.where(mask, img, out)


@dataclass
class Aug:
    crop: int = 128
    p_bin: float = 0.25
    # Downscale by a random factor in [1, max_scale] instead of exactly 2×.
    continuous: bool = False
    max_scale: float = 2.0
    p_blur: float = 0.25
    max_blur: float = 1.2
    p_bayer: float = 0.2
    # Shift each target patch so its median matches the input's: removes
    # what the stack matching left (1–6% of faint signal on M 17).
    offset_match: bool = False


def sample(pair: Pair, i: int, aug: Aug, rng: random.Random) -> tuple[np.ndarray, np.ndarray, float]:
    """(input, target, fwhm) in the network's domain, augmented identically.
    Inputs and targets swap at random (the pair is symmetric)."""
    a, b = pair.a[i], pair.b[i]
    if aug.offset_match:
        b = b + (np.median(a) - np.median(b))
    na, nb = pair.noise_a, pair.noise_b
    if rng.random() < 0.5:
        a, b, na, nb = b, a, nb, na
    fwhm = pair.fwhm
    ops = []
    if aug.continuous and rng.random() < aug.p_bin:
        f = rng.uniform(1.0, min(aug.max_scale, a.shape[0] / aug.crop))
        if f > 1.05:
            ops.append(lambda x, f=f: downscale(x, f))
            fwhm /= f
    elif not aug.continuous and rng.random() < aug.p_bin and a.shape[0] >= 2 * aug.crop:
        ops.append(bin2)
        fwhm /= 2
    if rng.random() < aug.p_blur:
        s = rng.uniform(0.4, aug.max_blur)
        ops.append(lambda x, s=s: blur(x, s))
        fwhm = math.hypot(fwhm, 2.3548 * s)
    if rng.random() < aug.p_bayer:
        c = rng.choice("rgb")
        ops.append(lambda x, c=c: bayer(x, c))
    if ops:
        before = noise_of(a)
        for op in ops:
            a, b = op(a), op(b)
        f = noise_of(a) / before if before > 0 else 1.0
        if np.isfinite(f) and f > 0:
            na, nb = na * f, nb * f
    h, w = a.shape
    c = min(aug.crop, h, w)
    y, x = rng.randrange(h - c + 1), rng.randrange(w - c + 1)
    a, b = a[y : y + c, x : x + c], b[y : y + c, x : x + c]
    k = rng.randrange(4)
    a, b = np.rot90(a, k), np.rot90(b, k)
    if rng.random() < 0.5:
        a, b = a[:, ::-1], b[:, ::-1]
    # Both in a's domain: the target keeps its own noise, scaled the same.
    xa = forward(a, pair.sky, na)
    xb = forward(b, pair.sky, na)
    return np.ascontiguousarray(xa, np.float32), np.ascontiguousarray(xb, np.float32), fwhm
