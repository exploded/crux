"""A small U-Net denoiser.

Input: [image in the asinh domain, FWHM / FWHM_SCALE, any further constant
control channels] (2 for the denoiser, 3 for sharpening). Output:
the denoised image in the same domain (input + learned residual). Sizes
must be multiples of 16; crux pads tiles.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

FWHM_SCALE = 4.0


def block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1),
        nn.LeakyReLU(0.1, inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1),
        nn.LeakyReLU(0.1, inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, base: int = 32, depth: int = 3, inputs: int = 2):
        super().__init__()
        ch = [base * 2**i for i in range(depth + 1)]
        self.down = nn.ModuleList()
        cin = inputs
        for c in ch[:-1]:
            self.down.append(block(cin, c))
            cin = c
        self.mid = block(ch[-2], ch[-1])
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        for c in reversed(ch[:-1]):
            self.up.append(nn.ConvTranspose2d(cin if self.up else ch[-1], c, 2, stride=2))
            self.dec.append(block(2 * c, c))
            cin = c
        self.out = nn.Conv2d(ch[0], 1, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        img = x[:, :1]
        skips = []
        h = x
        for d in self.down:
            h = d(h)
            skips.append(h)
            h = F.max_pool2d(h, 2)
        h = self.mid(h)
        for up, dec, s in zip(self.up, self.dec, reversed(skips)):
            h = up(h)
            h = dec(torch.cat([h, s], 1))
        return img + self.out(h)


def parameters(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
