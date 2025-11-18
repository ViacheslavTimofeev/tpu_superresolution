"""Minimal, production‑ready U‑Net (2D) starter in PyTorch.

Supports:
- Segmentation (logits per class)
- Image‑to‑image regression (e.g., denoising / simple SR head) via `out_channels=1` (or 3) and `loss=MSE/L1`
- Bilinear upsampling or transposed conv
- Optional BatchNorm, Dropout, and weight init

Usage examples are at the bottom (shape sanity check and parameter count).

Author: ChatGPT (starter template)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# Building blocks
# ----------------------------
def make_norm(norm: bool | str, num_ch: int):
    """
    norm=True  -> BatchNorm2d
    norm=False -> no norm
    norm='group32' -> GroupNorm(32)
    """
    if norm is True:
        return nn.BatchNorm2d(num_ch)
    if norm is False or norm is None:
        return None
    if isinstance(norm, str) and norm.startswith("group"):
        groups = int(norm.replace("group", ""))
        return nn.GroupNorm(groups, num_ch)
    raise ValueError(f"Unsupported norm spec: {norm}")

class ConvBNAct(nn.Sequential):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=None, groups=1,
                 norm: bool | str = True, act: bool = True, dropout: float = 0.0):
        if p is None:
            p = k // 2
        layers = [nn.Conv2d(in_ch, out_ch, k, s, p, bias=(norm is False),
                            groups=groups, padding_mode='reflect')]
        n = make_norm(norm, out_ch)
        if n is not None:
            layers.append(n)
        if act:
            layers.append(nn.ReLU(inplace=True))
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        super().__init__(*layers)
        
class SE(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        h = max(1, c // r)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(c, h, 1), nn.ReLU(inplace=True),
            nn.Conv2d(h, c, 1), nn.Sigmoid()
        )
    def forward(self, x):
        w = self.fc(self.pool(x))
        return x * w
        
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, mid_ch=None, norm=True, dropout=0.0):
        super().__init__()
        #self.se = SE(out_ch)
        #self.se = nn.Identity()
        if mid_ch is None:
            mid_ch = out_ch
        self.block = nn.Sequential(
            ConvBNAct(in_ch,  mid_ch, norm=norm),
            ConvBNAct(mid_ch, out_ch, norm=norm),
        )
        if dropout > 0:
            self.block.add_module("drop", nn.Dropout2d(dropout))

    def forward(self, x):
        x = self.block(x)
        #x = self.se(x)
        return x

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_ch: int, out_ch: int, norm: bool = True, dropout: float = 0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch, norm=norm, dropout=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Up(nn.Module):
    """Upscaling then double conv.
    
    Modes:
      - 'bilinear': Upsample + 1x1 conv (по умолчанию)
      - 'deconv'  : ConvTranspose2d
      - 'pixelshuffle': Conv -> PixelShuffle(r=2)
    """

    def __init__(self, in_ch: int, out_ch: int,
                 norm: bool = True, dropout: float = 0.0, up_mode: str = "pixelshuffle"):
        super().__init__()
        self.up_mode = up_mode
        mode = up_mode if up_mode is not None else "pixelshuffle"

        if mode == "bilinear":
            self.up = nn.Sequential(
                nn.Conv2d(in_ch, in_ch // 2, kernel_size=1),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            )
        elif mode == "deconv":
            self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        elif mode == "pixelshuffle":
            r = 2
            # делаем conv на низком разрешении и генерим r^2 * (in_ch//2) каналов
            self.up = nn.Sequential(
                nn.Conv2d(in_ch, (in_ch // 2) * (r * r), kernel_size=3, padding=1),
                nn.PixelShuffle(r),  # -> (in_ch//2, H*r, W*r)
            )
        else:
            raise ValueError(f"Unsupported up_mode: {mode}")

        # после апсемпла конкатинируем скип: каналы суммируются -> DoubleConv(in_ch, out_ch)
        self.conv = DoubleConv(in_ch, out_ch, norm=norm, dropout=dropout)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2], mode='reflect')
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# ----------------------------
# U‑Net
# ----------------------------
@dataclass
class UNetConfig:
    in_channels: int = 1
    out_channels: int = 1  # for segmentation: =num_classes; for regression: =1 or 3
    base_channels: int = 64
    depth: int = 4  # encoder depth; total levels = depth + 1 (including bottom)
    norm_enc: bool = True     # <-- раньше было norm: bool
    norm_dec: bool = True    # <-- отключим BN в декодере по умолчанию
    dropout: float = 0.0
    up_mode: str = "bilinear"  # <-- 'bilinear' | 'deconv' | 'pixelshuffle'


class UNet2D(nn.Module):
    """U‑Net (2D) starter.

    Args:
        cfg: UNetConfig with architecture hyperparameters.

    Notes:
        - Output logits have shape [B, out_channels, H, W]. Apply torch.sigmoid / softmax outside if needed.
        - Works with any input H, W (padding aligns skip connections).
    """

    def __init__(self, cfg: UNetConfig):
        super().__init__()
        c = cfg.base_channels
        self.cfg = cfg
        self.residual_scale = 0.1
        
        # Encoder
        self.inc = DoubleConv(cfg.in_channels, c, norm=cfg.norm_enc, dropout=cfg.dropout)
        self.downs = nn.ModuleList()
        ch = c
        for _ in range(cfg.depth):
            self.downs.append(Down(ch, ch * 2, norm=cfg.norm_enc, dropout=cfg.dropout))
            ch *= 2
        
        # Decoder
        self.ups = nn.ModuleList()
        for _ in range(cfg.depth):
            self.ups.append(Up(ch, ch // 2, norm=cfg.norm_dec, dropout=cfg.dropout, up_mode=cfg.up_mode))
            ch //= 2
        
        self.outc = OutConv(ch, cfg.out_channels)
        self._init_weights()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        skips = [x1]
        x_enc = x1
        for down in self.downs:
            x_enc = down(x_enc)
            skips.append(x_enc)

        x_dec = skips.pop()
        for up in self.ups:
            skip = skips.pop()
            x_dec = up(x_dec, skip)

        logits = self.outc(x_dec)
        return x + self.residual_scale * logits

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        # нулевой старт для residual-головы
        nn.init.zeros_(self.outc.conv.weight)
        if self.outc.conv.bias is not None:
            nn.init.zeros_(self.outc.conv.bias)
            
    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ----------------------------
# Quick sanity check
# ----------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = UNetConfig(in_channels=1, out_channels=2, base_channels=64, depth=4, norm=True, up_mode="bilinear")
    model = UNet2D(cfg).to(device)
    x = torch.randn(2, 1, 256, 256, device=device)
    y = model(x)
    print("Output:", y.shape)  # [2, 2, 256, 256]
    print("Params:", f"{model.n_params/1e6:.2f}M")

    # Example losses
    # Segmentation (2 classes):
    # logits = y; target = torch.randint(0, 2, (2, 256, 256), device=device)
    # loss = nn.CrossEntropyLoss()(logits, target)

    # Regression / restoration (e.g., denoise):
    # preds = y  # out_channels should match target channels
    # target = torch.randn_like(preds)
    # loss = nn.L1Loss()(preds, target)
