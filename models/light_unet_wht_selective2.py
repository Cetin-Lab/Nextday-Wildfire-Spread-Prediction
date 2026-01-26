# -*- coding: utf-8 -*-
"""
Updated LiteUNetSelectiveWHT with full WHT configuration
@author: jaych
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
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

        # FWHT transforms
        self.fwht_H = LearnableFWHTTransform(height, mode, learnable_T)
        self.fwht_W = LearnableFWHTTransform(width,  mode, learnable_T)

        # Trainable spectral scaling
        self.v = nn.Parameter(torch.ones(height, width))

        # Threshold in transform domain
        self.threshold = Thresholding(
            (height, width),
            mode=threshold_mode,
            init_scale=0.1,
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
        assert (H, W) == (self.H, self.W)

        X = self.forward_transform(x)
        X = X * self.v
        X = self.threshold(X)
        Y = self.inverse_transform(X)

        return Y

# ============================================================
# DoubleConv with optional WHT
# ============================================================
class DoubleConvWHT(nn.Module):
    def __init__(self, in_ch, out_ch, H, W, use_wht,
                 transform_mode, learnable_T, threshold_mode):
        super().__init__()
        self.use_wht = use_wht
        self.H, self.W = H, W
        self.relu = nn.ReLU(inplace=True)

        if use_wht:
            self.wht = WHT2D(H, W, transform_mode, learnable_T, threshold_mode)

            # "use the second conv for channel mapping": in_ch -> out_ch
            self.conv2 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
            self.bn2   = nn.BatchNorm2d(out_ch)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
            self.bn1   = nn.BatchNorm2d(out_ch)
            self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
            self.bn2   = nn.BatchNorm2d(out_ch)

    def forward(self, x):

        if self.use_wht:
            B, C, H, W = x.shape
            assert (H, W) == (self.H, self.W)

            # WHT per-channel
            x = x.reshape(B * C, 1, H, W)
            x = self.wht(x)
            x = x.reshape(B, C, H, W)

            # # (optional) explicit ReLU right after WHT
            # x = self.relu(x)

            # single conv block
            x = self.relu(self.bn2(self.conv2(x)))
            return x

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

# class DoubleConvWHT(nn.Module):
#     def __init__(self, in_ch, out_ch, H, W, use_wht,
#                  transform_mode, learnable_T, threshold_mode):
#         super().__init__()

#         self.use_wht = use_wht

#         if use_wht:
#             self.wht = WHT2D(
#                 height=H, width=W,
#                 mode=transform_mode,
#                 learnable_T=learnable_T,
#                 threshold_mode=threshold_mode
#             )
#   #         self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
#             self.bn1 = nn.BatchNorm2d(out_ch)

#         else:
#             self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
#             self.bn1 = nn.BatchNorm2d(out_ch)

#         self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(out_ch)
#         self.relu = nn.ReLU(inplace=True)

#         self.H, self.W = H, W

#     def forward(self, x):
#         if self.use_wht:
#             B, C, H, W = x.shape
        
#             # reshape handles non-contiguous tensors safely
#             x = x.reshape(B * C, 1, H, W)
        
#             x = self.wht(x)
        
#             x = x.reshape(B, C, H, W)

#         x = self.relu(self.bn1(self.conv1(x)))
#         x = self.relu(self.bn2(self.conv2(x)))
#         return x

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
# FULL UNET WITH WHT SELECTOR
# ============================================================

class LiteUNetSelectiveWHT(nn.Module):
    def __init__(
        self,
        n_channels=12,
        n_classes=1,
        base_c=4,
        bilinear=True,
        use_wht=None,
        transform_mode="walsh",
        learnable_T=False,
        threshold_mode="soft"
    ):
        super().__init__()

        if use_wht is None:
            use_wht = {k: False for k in ["inc","down1","down2","down3","down4"]}

        c1, c2, c3, c4, c5 = base_c, base_c*2, base_c*4, base_c*8, base_c*16
        factor = 2 if bilinear else 1

        # Encoder
        self.inc = DoubleConvWHT(
            n_channels, c1, 64, 64, use_wht["inc"],
            transform_mode, learnable_T, threshold_mode
        )

        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConvWHT(c1, c2, 32, 32, use_wht["down1"],
                          transform_mode, learnable_T, threshold_mode)
        )

        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConvWHT(c2, c3, 16, 16, use_wht["down2"],
                          transform_mode, learnable_T, threshold_mode)
        )

        self.down3 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConvWHT(c3, c4, 8, 8, use_wht["down3"],
                          transform_mode, learnable_T, threshold_mode)
        )

        self.down4 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConvWHT(c4, c5//factor, 4, 4, use_wht["down4"],
                          transform_mode, learnable_T, threshold_mode)
        )

        # Decoder
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

    use_wht = {
        "inc":   True,
        "down1": True,
        "down2":  True,
        "down3": True,
        "down4": True,
    }

    print("\n===== LiteUNetSelectiveWHT — Ablation Configuration =====")
    for k,v in use_wht.items():
        print(f" {k}: {'WHT2D' if v else 'Conv2D'}")
    print("==========================================================\n")

    model = LiteUNetSelectiveWHT(
        n_channels=12,
        n_classes=1,
        base_c=4,
        use_wht=use_wht,
        transform_mode="walsh",     # or "hadamard"
        learnable_T=False,          # set True to learn transform
        threshold_mode="soft"       # or "hard"
    )

    x = torch.randn(2, 12, 64, 64)
    y = model(x)
    for n,p in model.named_parameters():
        if "fwht_" in n or ".v" in n or "threshold.T" in n:
            print(n, p.requires_grad, p.shape)
    print("Input shape: ", x.shape)
    print("Output shape:", y.shape)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total params:     {total_params:,}")
    print(f"Trainable params: {trainable_params:,}\n")

