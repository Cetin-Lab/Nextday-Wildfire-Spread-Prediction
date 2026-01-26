# -*- coding: utf-8 -*-
"""
Created on Fri Jan  9 13:29:28 2026

@author: olivi
"""
# -*- coding: utf-8 -*-
"""
LiteUNetSelectiveDCT (OPTIONAL COMPRESSION)
- Replaces the WHT2D (Hadamard/Walsh) block with a 2D DCT block.
- Compression (low-frequency cropping) is OPTIONAL:
    * If compression_ratio >= 1.0 (default) -> no cropping (full spectrum).
    * If 0 < compression_ratio < 1.0      -> keep top-left (ch x cw) DCT coefficients,
                                            apply scaling + threshold there, then zero-pad.

Interface:
  use_dct = {"inc":True/False, "down1":..., ...}
  compression_ratio: float in (0,1], applied to all DCT blocks
    - set to 1.0 to disable compression (default)
    - set to e.g. 0.5 to keep low-frequency 50% block
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ============================================================
# DCT matrix builder + cache (pure torch, orthonormal DCT-II)
# ============================================================

_DCT_CACHE = {}  # key: (N, device, dtype) -> C (NxN)

def dct_ortho_matrix(N: int, device, dtype):
    """
    Orthonormal DCT-II matrix C (N x N):
      y = x @ C.T  applies DCT-II along last dimension
      x = y @ C    inverse (since C is orthonormal)
    """
    key = (N, str(device), str(dtype))
    if key in _DCT_CACHE:
        return _DCT_CACHE[key]

    n = torch.arange(N, device=device, dtype=dtype).view(1, N)  # (1,N)
    k = torch.arange(N, device=device, dtype=dtype).view(N, 1)  # (N,1)

    C = torch.cos(torch.pi * (n + 0.5) * k / N)                 # (N,N)
    alpha = torch.ones((N, 1), device=device, dtype=dtype) * math.sqrt(2.0 / N)
    alpha[0, 0] = math.sqrt(1.0 / N)
    C = C * alpha

    _DCT_CACHE[key] = C
    return C

# ============================================================
# Threshold Layer
# ============================================================

class Thresholding(nn.Module):
    def __init__(self, t_shape, mode="soft", init_scale=0.1, learnable=True, ste_for_hard=True):
        super().__init__()
        self.mode = mode
        self.ste = ste_for_hard
        T = torch.rand(t_shape) * init_scale
        self.T = nn.Parameter(T, requires_grad=learnable)

    def forward(self, x):
        Tabs = self.T.abs()
        ax = x.abs()

        if self.mode == "soft":
            return torch.sign(x) * F.relu(ax - Tabs)

        elif self.mode == "hard":
            mask = (ax > Tabs).float()
            y = x * mask
            return y + (x - y).detach() if self.ste else y

        else:
            raise ValueError("unknown threshold mode")

# ============================================================
# DCT2D Block with OPTIONAL COMPRESSION
# ============================================================

class DCT2D(nn.Module):
    """
    2D DCT block:
      X = DCT2(x)
      if compression enabled:
          X_low = X[:ch, :cw]
          X_low = v ● X_low
          X_low = threshold(X_low)
          X_pad = pad(X_low -> HxW)
          Y = IDCT2(X_pad)
      else:
          X = v ● X
          X = threshold(X)
          Y = IDCT2(X)
    """

    def __init__(self, height, width, compression_ratio=1.0, learnable_T=False, threshold_mode="soft"):
        super().__init__()
        self.H = int(height)
        self.W = int(width)

        cr = float(compression_ratio)
        if not (0.0 < cr <= 1.0):
            raise ValueError("compression_ratio must be in (0, 1]. Use 1.0 to disable compression.")
        self.compression_ratio = cr
        self.use_compression = (cr < 1.0)

        if self.use_compression:
            self.ch = max(1, int(self.H * cr))
            self.cw = max(1, int(self.W * cr))
        else:
            self.ch = self.H
            self.cw = self.W

        self.learnable_T = bool(learnable_T)
        self._initialized_T = False

        # Spectral scaling + threshold shape follows (ch, cw)
        self.v = nn.Parameter(torch.ones(self.ch, self.cw))
        self.threshold = Thresholding(
            (self.ch, self.cw),
            mode=threshold_mode,
            init_scale=0.1,
            learnable=True
        )

    def _init_mats(self, device, dtype):
        C_H = dct_ortho_matrix(self.H, device=device, dtype=dtype)
        C_W = dct_ortho_matrix(self.W, device=device, dtype=dtype)

        if self.learnable_T:
            self.C_H = nn.Parameter(C_H.clone(), requires_grad=True)
            self.C_W = nn.Parameter(C_W.clone(), requires_grad=True)
        else:
            self.register_buffer("C_H_buf", C_H, persistent=False)
            self.register_buffer("C_W_buf", C_W, persistent=False)

        self._initialized_T = True

    def _get_CH_CW(self):
        if self.learnable_T:
            return self.C_H, self.C_W
        return self.C_H_buf, self.C_W_buf

    def forward_transform(self, x):
        C_H, C_W = self._get_CH_CW()
        x = x @ C_W.T
        x = x.transpose(-1, -2)     # (B,C,W,H)
        x = x @ C_H.T
        x = x.transpose(-1, -2)     # (B,C,H,W)
        return x

    def inverse_transform(self, x):
        C_H, C_W = self._get_CH_CW()
        x = x.transpose(-1, -2)     # (B,C,W,H)
        x = x @ C_H
        x = x.transpose(-1, -2)     # (B,C,H,W)
        x = x @ C_W
        return x

    def forward(self, x):
        if not self._initialized_T:
            self._init_mats(device=x.device, dtype=x.dtype)

        B, C, H, W = x.shape
        assert (H, W) == (self.H, self.W), f"Expected {(self.H,self.W)} got {(H,W)}"

        X = self.forward_transform(x)

        if self.use_compression:
            X_low = X[..., :self.ch, :self.cw]
            X_low = X_low * self.v
            X_low = self.threshold(X_low)
            X_proc = F.pad(X_low, pad=(0, self.W - self.cw, 0, self.H - self.ch))
        else:
            X_full = X * self.v
            X_proc = self.threshold(X_full)

        Y = self.inverse_transform(X_proc)
        return Y

# ============================================================
# DoubleConv with optional DCT
# ============================================================

class DoubleConvDCT(nn.Module):
    def __init__(self, in_ch, out_ch, H, W, use_dct,
                 compression_ratio=1.0, learnable_T=False, threshold_mode="soft"):
        super().__init__()
        self.use_dct = use_dct
        self.H, self.W = H, W
        self.relu = nn.ReLU(inplace=True)

        if use_dct:
            self.dct = DCT2D(
                height=H, width=W,
                compression_ratio=compression_ratio,
                learnable_T=learnable_T,
                threshold_mode=threshold_mode
            )

            # Use conv2 for channel mapping: in_ch → out_ch
            self.conv2 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
            self.bn2   = nn.BatchNorm2d(out_ch)

        else:
            # Standard DoubleConv
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
            self.bn1   = nn.BatchNorm2d(out_ch)

            self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
            self.bn2   = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        if self.use_dct:
            B, C, H, W = x.shape
            assert (H, W) == (self.H, self.W), f"Expected {(self.H,self.W)}, got {(H,W)}"

            # Apply DCT per-channel
            x = x.reshape(B * C, 1, H, W)
            x = self.dct(x)
            x = x.reshape(B, C, H, W)

            # # Optional ReLU after DCT (matches your WHT design)
            # x = self.relu(x)

            # Single 3×3 conv for channel mapping
            x = self.relu(self.bn2(self.conv2(x)))
            return x

        # Non-DCT path: standard DoubleConv
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

# ============================================================
# Standard UNet Components (unchanged)
# ============================================================

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)

class Up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_ch, out_ch)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)
            self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diffX//2, diffX - diffX//2,
                        diffY//2, diffY - diffY//2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        return self.conv(x)

# ============================================================
# FULL UNET WITH DCT SELECTOR + OPTIONAL COMPRESSION
# ============================================================

class LiteUNetSelectiveDCT(nn.Module):
    def __init__(
        self,
        n_channels=12,
        n_classes=1,
        base_c=4,
        bilinear=True,
        use_dct=None,
        compression_ratio=1.0,   # 1.0 disables compression (default)
        learnable_T=False,
        threshold_mode="soft"
    ):
        super().__init__()

        if use_dct is None:
            use_dct = {k: False for k in ["inc","down1","down2","down3","down4"]}

        c1, c2, c3, c4, c5 = base_c, base_c*2, base_c*4, base_c*8, base_c*16
        factor = 2 if bilinear else 1

        self.inc = DoubleConvDCT(
            n_channels, c1, 64, 64, use_dct["inc"],
            compression_ratio=compression_ratio,
            learnable_T=learnable_T,
            threshold_mode=threshold_mode
        )

        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConvDCT(c1, c2, 32, 32, use_dct["down1"],
                          compression_ratio=compression_ratio,
                          learnable_T=learnable_T,
                          threshold_mode=threshold_mode)
        )

        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConvDCT(c2, c3, 16, 16, use_dct["down2"],
                          compression_ratio=compression_ratio,
                          learnable_T=learnable_T,
                          threshold_mode=threshold_mode)
        )

        self.down3 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConvDCT(c3, c4, 8, 8, use_dct["down3"],
                          compression_ratio=compression_ratio,
                          learnable_T=learnable_T,
                          threshold_mode=threshold_mode)
        )

        self.down4 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConvDCT(c4, c5//factor, 4, 4, use_dct["down4"],
                          compression_ratio=compression_ratio,
                          learnable_T=learnable_T,
                          threshold_mode=threshold_mode)
        )

        self.up1 = Up(c5//factor + c4, c4//factor, bilinear)
        self.up2 = Up(c4//factor + c3, c3//factor, bilinear)
        self.up3 = Up(c3//factor + c2, c2//factor, bilinear)
        self.up4 = Up(c2//factor + c1, c1, bilinear)

        self.outc = OutConv(c1, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)

# ============================================================
# Standalone test
# ============================================================

if __name__ == "__main__":
    use_dct = {"inc": True, "down1": True, "down2": True, "down3": True, "down4": True}

    print("\n===== LiteUNetSelectiveDCT (Optional Compression) — Ablation Configuration =====")
    for k, v in use_dct.items():
        print(f" {k}: {'DCT2D' if v else 'Conv2D'}")
    print("==========================================================\n")

    model = LiteUNetSelectiveDCT(
        n_channels=12,
        n_classes=1,
        base_c=4,
        use_dct=use_dct,
        compression_ratio=0.5,   # set 1.0 to disable compression
        learnable_T=False,
        threshold_mode="soft"
    )

    x = torch.randn(2, 12, 64, 64)
    y = model(x)
    for n,p in model.named_parameters():
        if ".v" in n or "threshold.T" in n or "C_H" in n or "C_W" in n:
            print(n, p.requires_grad, p.shape)

    print("Input shape: ", x.shape)
    print("Output shape:", y.shape)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total params:     {total_params:,}")
    print(f"Trainable params: {trainable_params:,}\n")
