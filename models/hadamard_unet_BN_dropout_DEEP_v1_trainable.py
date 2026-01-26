# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 17:31:37 2025

@author: olivi
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.linalg import hadamard
from torchsummary import summary


# ==========================================================
# 🔹 Orthogonal Learnable Hadamard Transform (QR-based)
# ==========================================================
class OrthoHadamard(nn.Module):
    """
    Learnable Hadamard-like transform with QR-based orthogonalization.
    Starts from the classic Hadamard basis but learns small rotations.
    """
    def __init__(self, n):
        super().__init__()
        H_init = torch.tensor(hadamard(n), dtype=torch.float32)
        self.W = nn.Parameter(H_init + 0.01 * torch.randn_like(H_init))

    def forward(self, x, axis=-1):
        # Reorthogonalize via QR decomposition each forward pass
        Q, _ = torch.linalg.qr(self.W)
        if axis != -1:
            x = torch.transpose(x, -1, axis)
        y = x @ Q
        if axis != -1:
            y = torch.transpose(y, -1, axis)
        return y

    def inverse(self, x, axis=-1):
        Q, _ = torch.linalg.qr(self.W)
        if axis != -1:
            x = torch.transpose(x, -1, axis)
        y = x @ Q.T
        if axis != -1:
            y = torch.transpose(y, -1, axis)
        return y / x.shape[-1]  # normalize energy


# ==========================================================
# 🔹 Trainable Thresholding Layers
# ==========================================================
class SoftThresholding(nn.Module):
    """Soft thresholding (trainable per feature)."""
    def __init__(self, dim):
        super().__init__()
        self.T = nn.Parameter(torch.rand(dim) / 10)

    def forward(self, x):
        return torch.sign(x) * F.relu(torch.abs(x) - torch.abs(self.T))


class HardThresholding(nn.Module):
    """Hard thresholding (trainable per feature)."""
    def __init__(self, dim):
        super().__init__()
        self.T = nn.Parameter(torch.rand(dim) / 10)

    def forward(self, x):
        mask = (torch.abs(x) > torch.abs(self.T)).float()
        return x * mask


# ==========================================================
# 🔹 Hadamard UNet (with Learnable Transform + Thresholds)
# ==========================================================
class HadamardUnet(nn.Module):
    def __init__(self, input_channels=12, input_size=64, output_channels=1, dropout_rate=0.3, threshold_mode="soft"):
        super().__init__()
        self.height = input_size // 2
        self.width  = input_size // 2
        C = 4  # feature channels in HT blocks

        # --- Convolutional Encoder ---
        self.conv1 = nn.Conv2d(input_channels, C, kernel_size=4, stride=2, padding=1)
        self.bn1   = nn.BatchNorm2d(C)
        self.dropout1 = nn.Dropout2d(dropout_rate)

        self.conv2 = nn.Conv2d(C, C, kernel_size=7, stride=1, padding=3)
        self.bn2   = nn.BatchNorm2d(C)
        self.dropout2 = nn.Dropout2d(dropout_rate)

        self.conv3 = nn.Conv2d(C, C, kernel_size=7, stride=1, padding=3)
        self.bn3   = nn.BatchNorm2d(C)
        self.dropout3 = nn.Dropout2d(dropout_rate)

        # --- Decoder ---
        self.deconv1 = nn.ConvTranspose2d(C, C, kernel_size=7, stride=1, padding=3)
        self.bn4     = nn.BatchNorm2d(C)
        self.dropout4 = nn.Dropout2d(dropout_rate)

        self.deconv2 = nn.ConvTranspose2d(C, C, kernel_size=7, stride=1, padding=3)
        self.bn5     = nn.BatchNorm2d(C)
        self.dropout5 = nn.Dropout2d(dropout_rate)

        self.deconv3 = nn.ConvTranspose2d(C, output_channels, kernel_size=4, stride=2, padding=1)

        # --- Learnable Hadamard Transforms ---
        self.HT_h = OrthoHadamard(self.height)
        self.HT_w = OrthoHadamard(self.width)

        # --- Trainable Thresholding Layers ---
        Thr = SoftThresholding if threshold_mode == "soft" else HardThresholding
        self.ST1 = Thr((1, C, self.height, self.width))
        self.ST2 = Thr((1, C, self.height, self.width))
        self.ST3 = Thr((1, C, self.height, self.width))
        self.ST4 = Thr((1, C, self.height, self.width))
        self.ST5 = Thr((1, C, self.height, self.width))

        # --- Per-channel learnable scaling masks ---
        self.v1 = nn.Parameter(torch.rand(1, C, self.height, self.width))
        self.v2 = nn.Parameter(torch.rand(1, C, self.height, self.width))
        self.v3 = nn.Parameter(torch.rand(1, C, self.height, self.width))
        self.v4 = nn.Parameter(torch.rand(1, C, self.height, self.width))
        self.v5 = nn.Parameter(torch.rand(1, C, self.height, self.width))

    def forward(self, x):
        # ========== Encoder ==========
        x1 = F.relu(self.bn1(self.conv1(x)))
        x1 = self.dropout1(x1)

        x2 = self.HT_h.forward(self.HT_w.forward(x1, axis=-1), axis=-2)
        x3 = self.v1 * x2
        x4 = self.ST1(x3)
        x5 = self.HT_w.inverse(self.HT_h.inverse(x4, axis=-2), axis=-1)

        # Block 2
        x6 = F.relu(self.bn2(self.conv2(x5)))
        x6 = self.dropout2(x6)

        x7  = self.HT_h.forward(self.HT_w.forward(x6, axis=-1), axis=-2)
        x8  = self.v2 * x7
        x9  = self.ST2(x8)
        x10 = self.HT_w.inverse(self.HT_h.inverse(x9, axis=-2), axis=-1)

        # Block 3
        x11 = F.relu(self.bn3(self.conv3(x10)))
        x11 = self.dropout3(x11)

        x12 = self.HT_h.forward(self.HT_w.forward(x11, axis=-1), axis=-2)
        x13 = self.v3 * x12
        x14 = self.ST3(x13)
        x15 = self.HT_w.inverse(self.HT_h.inverse(x14, axis=-2), axis=-1)

        # ========== Decoder ==========
        x16 = F.relu(self.bn4(self.deconv1(x15)))
        x16 = self.dropout4(x16)
        x16 = x16 + x11  # feature-space skip

        x17 = self.HT_h.forward(self.HT_w.forward(x16, axis=-1), axis=-2)
        x18 = self.v4 * x17
        x19 = self.ST4(x18) + x9  # residual in HT domain
        x20 = self.HT_w.inverse(self.HT_h.inverse(x19, axis=-2), axis=-1)

        x21 = F.relu(self.bn5(self.deconv2(x20)))
        x21 = self.dropout5(x21)
        x21 = x21 + x6  # feature-space skip

        x22 = self.HT_h.forward(self.HT_w.forward(x21, axis=-1), axis=-2)
        x23 = self.v5 * x22
        x24 = self.ST5(x23) + x4  # residual in HT domain
        x25 = self.HT_w.inverse(self.HT_h.inverse(x24, axis=-2), axis=-1)

        # ========== Output ==========
        x_out = self.deconv3(x25)
        return x_out


# ==========================================================
# 🔹 Sanity Check / Summary
# ==========================================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HadamardUnet(input_channels=12, input_size=64, output_channels=1, threshold_mode="soft").to(device)
    summary(model, input_size=(12, 64, 64), device=str(device))

    dummy = torch.randn(2, 12, 64, 64, device=device)
    with torch.no_grad():
        out = model(dummy)
    print("✅ Output shape:", out.shape)
