"""Check crux's Rust inference against PyTorch on one bundle patch.

    python tools/ml/check_rust.py --run .local/ml/runs/r1 --bundle <bundle> --crux <crux.exe with ml>
"""

from __future__ import annotations

import argparse
import struct
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch

import bundles as B
from model import UNet
from train import denoise


def write_fits(path: Path, img: np.ndarray):
    h, w = img.shape
    cards = ["SIMPLE  =                    T", "BITPIX  =                  -32", "NAXIS   =                    2",
             f"NAXIS1  = {w:20d}", f"NAXIS2  = {h:20d}", "END"]
    hdr = "".join(c.ljust(80) for c in cards).encode("ascii")
    hdr += b" " * (-len(hdr) % 2880)
    data = img.astype(">f4").tobytes()
    data += b"\0" * (-len(data) % 2880)
    path.write_bytes(hdr + data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--crux", type=Path, required=True)
    args = ap.parse_args()
    ck = torch.load(args.run / "model.pt", map_location="cpu")
    net = UNet(base=ck.get("base", 32), depth=ck.get("depth", 3)).eval()
    net.load_state_dict(ck["model"])
    p = B.load_bundle(args.bundle)[0]
    a = p.a[0]
    want = denoise(net, a, p.sky, p.noise_a, p.fwhm, torch.device("cpu"))
    with tempfile.TemporaryDirectory() as d:
        src, out = Path(d) / "patch.fits", Path(d) / "out.fits"
        write_fits(src, a)
        subprocess.run([str(args.crux), "dev", "denoise", str(src), "--model", str(args.run / "denoise.onnx"), "--out", str(out),
                        "--tile", str(a.shape[0]), "--noise", repr(p.noise_a), "--sky", repr(p.sky), "--fwhm", repr(p.fwhm)], check=True)
        _, got = B.read_fits_cube(out)
    got = got[0]
    diff = np.abs(got - want)
    print(f"patch {a.shape}, noise {p.noise_a:.3e}: max |Rust - PyTorch| {diff.max():.3e} "
          f"({diff.max() / p.noise_a:.2e} sigma), mean {diff.mean() / p.noise_a:.2e} sigma; "
          f"denoising changed pixels by {np.abs(want - a).mean() / p.noise_a:.2f} sigma on average")


if __name__ == "__main__":
    main()
