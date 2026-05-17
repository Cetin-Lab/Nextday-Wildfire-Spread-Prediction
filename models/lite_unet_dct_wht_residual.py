# -*- coding: utf-8 -*-
"""
LiteUNetDCTWHTResidual (UPDATED)
✅ Supports input_size = 64, 128, 256, ... (power of 2)
✅ WHT2D sizes automatically derived from input_size
✅ Keeps exact WHT correctness (no resizing hacks)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ============================================================
# Accurate Butterfly-Based FWHT and iFWHT
# ============================================================

def fwht_butterfly(x):
    N = x.shape[-1]
    assert (N & (N - 1)) == 0, "Input length must be power of 2"
    x = x.clone()
    h = 1
    while h < N:
        for i in range(0, N, h * 2):
            for j in range(i, i + h):
                a = x[..., j].clone()
                b = x[..., j + h].clone()
                x[..., j]     = a + b
                x[..., j + h] = a - b
        h *= 2
    return x


def ifwht_butterfly(x):
    x = fwht_butterfly(x)
    return x / x.shape[-1]

# ============================================================
# Hadamard → Walsh ordering
# ============================================================

def hadamard_to_walsh(H):
    sign_changes = torch.sum(H[:, :-1] * H[:, 1:] < 0, dim=1)
    return H[torch.argsort(sign_changes)]

# ============================================================
# Learnable FWHT Transform
# ============================================================

class LearnableFWHTTransform(nn.Module):
    def __init__(self, size, mode="hadamard", learnable=False, device="cpu"):
        super().__init__()
        assert (size & (size - 1)) == 0

        I = torch.eye(size, device=device)
        T = fwht_butterfly(I)

        if mode.lower() == "walsh":
            T = hadamard_to_walsh(T)

        self.T = nn.Parameter(T, requires_grad=learnable)
        self.size = size

    def forward(self, x, axis=-1, inverse=False):
        T = self.T.T if not inverse else self.T
        x = x.transpose(axis, -1)
        x = x @ T
        x = x.transpose(-1, axis)
        return x / self.size if inverse else x

# ============================================================
# Threshold Layer
# ============================================================

class Thresholding(nn.Module):
    def __init__(self, t_shape, mode="soft", init_scale=0.1, learnable=True, ste_for_hard=True):
        super().__init__()
        self.mode = mode
        self.ste = ste_for_hard
        T = torch.rand(t_shape) * init_scale
        self.T = nn.Parameter(T) if learnable else nn.Parameter(T, requires_grad=False)

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
# DCT matrix builder + cache (pure torch, orthonormal DCT-II)
# ============================================================

_DCT_CACHE = {}  # key: (N, device, dtype) -> C (NxN)

def dct_ortho_matrix(N: int, device, dtype):
    """
    Orthonormal DCT-II matrix C (N x N):
      y = x @ C.T  applies DCT-II along last dimension
      x = y @ C    inverse (since C is orthonormal)
    """
    key = (int(N), str(device), str(dtype))
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
# DCT2D Block (OPTIONAL COMPRESSION)
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
            init_scale=0.01,
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
# WHT2D Block
# ============================================================

class WHT2D(nn.Module):
    """
    2D Walsh-Hadamard block:
      X = WHT(x)
      X = v ● X
      X = threshold(X)
      x_hat = iWHT(X)
    """

    def __init__(self, height, width, mode, learnable_T, threshold_mode):
        super().__init__()

        assert (height & (height - 1)) == 0
        assert (width  & (width  - 1)) == 0

        self.H = height
        self.W = width

        self.fwht_H = LearnableFWHTTransform(height, mode, learnable_T)
        self.fwht_W = LearnableFWHTTransform(width,  mode, learnable_T)

        self.v = nn.Parameter(torch.ones(height, width))

        self.threshold = Thresholding(
            (height, width),
            mode=threshold_mode,
            init_scale=0.01,
            learnable=True
        )

    def forward_transform(self, x):
        x = self.fwht_W(x, axis=3)
        x = self.fwht_H(x, axis=2)
        return x

    def inverse_transform(self, x):
        x = self.fwht_H(x, axis=2, inverse=True)
        x = self.fwht_W(x, axis=3, inverse=True)
        return x

    def forward(self, x):
        B, C, H, W = x.shape
        assert (H, W) == (self.H, self.W), f"Expected {(self.H,self.W)}, got {(H,W)}"

        X = self.forward_transform(x)
        X = X * self.v
        X = self.threshold(X)
        Y = self.inverse_transform(X)

        return Y

# ============================================================
# DoubleConv with optional WHT
# ============================================================
class SpectralFusion(nn.Module):
    def __init__(self, ch, r=8):
        super().__init__()
        hidden = max(4, ch // r)
        self.mlp = nn.Sequential(
            nn.Linear(2 * ch, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, ch),
        )

    def forward(self, x_wht, x_dct):
        B, C, _, _ = x_wht.shape
        g1 = x_wht.mean(dim=(2,3))
        g2 = x_dct.mean(dim=(2,3))
        w = torch.sigmoid(self.mlp(torch.cat([g1, g2], dim=1))).view(B, C, 1, 1)
        return w * x_wht + (1 - w) * x_dct




class DoubleConvWHT(nn.Module):
    def __init__(self, in_ch, out_ch, H, W, use_wht,
                 transform_mode, learnable_T, threshold_mode, dct_compression_ratio=1.0, use_residual=False):
        super().__init__()
        self.use_wht = use_wht
        self.H, self.W = H, W
        self.use_residual = use_residual
        self.fuse = SpectralFusion(in_ch)
        
        if use_wht:
            self.wht = WHT2D(
                height=H, width=W,
                mode=transform_mode,
                learnable_T=learnable_T,
                threshold_mode=threshold_mode
            )
            self.dct = DCT2D(
                height=H, width=W,
                compression_ratio=dct_compression_ratio,
                learnable_T=learnable_T,
                threshold_mode=threshold_mode
            )

            self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
            self.bn1 = nn.BatchNorm2d(out_ch)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(out_ch)

        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        debug_outputs = {}
        if self.use_wht:
            B, C, H, W = x.shape
            assert (H, W) == (self.H, self.W), f"WHT/DCT expects {(self.H,self.W)}, got {(H,W)}"

            # Apply both transforms per-channel, then concat (B, 2C, H, W)
            x_flat = x.reshape(B * C, 1, H, W)
            x_wht = self.wht(x_flat).reshape(B, C, H, W)
            x_dct = self.dct(x_flat).reshape(B, C, H, W)
            x = self.fuse(x_wht, x_dct) + x if self.use_residual else self.fuse(x_wht, x_dct)

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

# ============================================================
# Standard UNet Components
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
# FULL UNET WITH WHT SELECTOR (UPDATED FOR input_size)
# ============================================================

class LiteUNetDCTWHTResidual(nn.Module):
    def __init__(
        self,
        n_channels=12,
        n_classes=1,
        base_c=4,
        bilinear=True,
        use_wht=None,
        input_size=64,                 # ✅ NEW: supports 64/128/256...
        transform_mode="walsh",
        learnable_T=False,
        threshold_mode="soft",
        dct_compression_ratio=0.7
    ):
        super().__init__()

        assert (input_size & (input_size - 1)) == 0, "input_size must be power of 2"
        assert input_size % 16 == 0, "input_size must be divisible by 16 (4 downsamples)"
        self.input_size = input_size

        if use_wht is None:
            use_wht = {k: False for k in ["inc","down1","down2","down3","down4"]}

        # Spatial sizes at each stage
        s0 = input_size
        s1 = s0 // 2
        s2 = s1 // 2
        s3 = s2 // 2
        s4 = s3 // 2

        c1, c2, c3, c4, c5 = base_c, base_c*2, base_c*4, base_c*8, base_c*16
        factor = 2 if bilinear else 1

        # Encoder
        self.inc = DoubleConvWHT(
            n_channels, c1, s0, s0, use_wht["inc"],
            transform_mode, learnable_T, threshold_mode,
            dct_compression_ratio=dct_compression_ratio
        )

        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConvWHT(c1, c2, s1, s1, use_wht["down1"],
                          transform_mode, learnable_T, threshold_mode, dct_compression_ratio=dct_compression_ratio,use_residual=False)
        )

        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConvWHT(c2, c3, s2, s2, use_wht["down2"],
                          transform_mode, learnable_T, threshold_mode, dct_compression_ratio=dct_compression_ratio,use_residual=True)
        )

        self.down3 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConvWHT(c3, c4, s3, s3, use_wht["down3"],
                          transform_mode, learnable_T, threshold_mode, dct_compression_ratio=dct_compression_ratio,use_residual=False)
        )

        self.down4 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConvWHT(c4, c5//factor, s4, s4, use_wht["down4"],
                          transform_mode, learnable_T, threshold_mode, dct_compression_ratio=dct_compression_ratio,use_residual=True)
        )

        # Decoder
        self.up1 = Up(c5//factor + c4, c4//factor, bilinear)
        self.up2 = Up(c4//factor + c3, c3//factor, bilinear)
        self.up3 = Up(c3//factor + c2, c2//factor, bilinear)
        self.up4 = Up(c2//factor + c1, c1, bilinear)

        self.outc = OutConv(c1, n_classes)

    def forward(self, x):
        # x: (B, n_channels, input_size, input_size)
        assert x.shape[-1] == self.input_size and x.shape[-2] == self.input_size, \
            f"Expected {self.input_size}×{self.input_size}, got {x.shape[-2:]}"

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

