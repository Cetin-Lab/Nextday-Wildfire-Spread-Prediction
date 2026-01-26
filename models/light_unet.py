# -*- coding: utf-8 -*-
"""
Created on Tue Jan 13 12:34:02 2026

@author: olivi
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Building blocks
# -----------------------------
class DoubleConv(nn.Module):
    """(Conv -> BN -> ReLU) * 2"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    """Downscale: MaxPool -> DoubleConv"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    """Upscale then DoubleConv (with skip concat)"""
    def __init__(self, in_ch, out_ch, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            # After concat, in_ch = ch_from_skip + ch_from_up
            self.conv = DoubleConv(in_ch, out_ch)
        else:
            # You can switch to transpose conv if you like
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        # x1: decoder feature, x2: encoder skip
        x1 = self.up(x1)
        # handle possible size mismatch due to pooling/upsample
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # concat skip
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


# -----------------------------
# Lightweight full UNet
# -----------------------------
class LiteUNet(nn.Module):
    """
    Full UNet (multi-stage encoder-decoder) but lightweight.

    - base_c controls width:
        base_c = 4  -> ~53k params (very light, close to HT-Unet)
        base_c = 8  -> ~211k params
    """
    def __init__(self, n_channels=12, n_classes=1, base_c=4, bilinear=True):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        c1 = base_c
        c2 = base_c * 2
        c3 = base_c * 4
        c4 = base_c * 8
        c5 = base_c * 16

        factor = 2 if bilinear else 1  # classic trick from UNet repo

        # Encoder
        self.inc   = DoubleConv(n_channels, c1)           # 64x64
        self.down1 = Down(c1, c2)                         # 32x32
        self.down2 = Down(c2, c3)                         # 16x16
        self.down3 = Down(c3, c4)                         # 8x8
        self.down4 = Down(c4, c5 // factor)               # 4x4 (bottleneck)

        # Decoder
        self.up1 = Up(c5 // factor + c4, c4 // factor, bilinear)  # 8x8
        self.up2 = Up(c4 // factor + c3, c3 // factor, bilinear)  # 16x16
        self.up3 = Up(c3 // factor + c2, c2 // factor, bilinear)  # 32x32
        self.up4 = Up(c2 // factor + c1, c1,           bilinear)  # 64x64

        self.outc = OutConv(c1, n_classes)

    def forward(self, x):
        x1 = self.inc(x)      # 64x64
        x2 = self.down1(x1)   # 32x32
        x3 = self.down2(x2)   # 16x16
        x4 = self.down3(x3)   # 8x8
        x5 = self.down4(x4)   # 4x4 (bottleneck)

        x = self.up1(x5, x4)  # 8x8
        x = self.up2(x,  x3)  # 16x16
        x = self.up3(x,  x2)  # 32x32
        x = self.up4(x,  x1)  # 64x64

        logits = self.outc(x) # (B, 1, 64, 64)
        return logits
# if __name__ == "__main__":
#     model = LiteUNet(n_channels=12, n_classes=1, base_c=4)  # try 4 or 8
#     x = torch.randn(2, 12, 64, 64)
#     y = model(x)
#     print("Output shape:", y.shape)

#     total = sum(p.numel() for p in model.parameters())
#     print("Total params:", total)
