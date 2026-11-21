"""Export a training checkpoint (model.pt) to ONNX with its JSON, for crux.

    python export.py <run>/model.pt --kind starless --out <dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from model import FWHM_SCALE, UNet

KINDS = {
    "denoise": ("crux-denoise", ["asinh((v - sky) / (k * noise))", "fwhm / fwhm_scale"]),
    "sharpen": ("crux-sharpen", ["asinh((v - sky) / (k * noise))", "fwhm / fwhm_scale", "sharpen (0 = correct only, 1 = also tighten)"]),
    "starless": ("crux-starless", ["asinh((v - sky) / (k * noise))", "fwhm / fwhm_scale"]),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--kind", choices=list(KINDS), required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    ck = torch.load(args.checkpoint, map_location="cpu")
    fmt, inputs = KINDS[args.kind]
    net = UNet(base=ck.get("base", 32), depth=ck.get("depth", 3), inputs=ck.get("inputs", len(inputs))).eval()
    net.load_state_dict(ck["model"])
    args.out.mkdir(parents=True, exist_ok=True)
    x = torch.zeros(1, len(inputs), 64, 64)
    onnx = args.out / f"{args.kind}.onnx"
    torch.onnx.export(net, x, onnx, input_names=["input"], output_names=["output"],
                      dynamic_axes={"input": {2: "h", 3: "w"}, "output": {2: "h", 3: "w"}}, opset_version=17, dynamo=False)
    meta = {"format": fmt, "version": 1, "k": 3.0, "fwhm_scale": FWHM_SCALE, "multiple": 2 ** ck.get("depth", 3) * 2,
            "inputs": inputs, "iter": ck.get("iter")}
    (args.out / f"{args.kind}.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print("exported", onnx)


if __name__ == "__main__":
    main()
