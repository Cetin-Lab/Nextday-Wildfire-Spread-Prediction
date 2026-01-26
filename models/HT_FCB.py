# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 17:26:57 2025

@author: olivi
"""
# -*- coding: utf-8 -*-
"""
Hadamard Residual Net (HT-FCB) with Auto Channel Detection
Author: Shuaiang Rong
Updated: Oct 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.linalg import hadamard


# ==========================================================
# 🔹 Walsh–Hadamard Transform Utilities
# ==========================================================
def fwht(x, axis=-1):
    """Forward Walsh–Hadamard Transform."""
    if axis != -1:
        x = torch.transpose(x, -1, axis)
    n = x.shape[-1]
    assert (n & (n - 1) == 0), "Length must be power of 2"
    H = torch.tensor(hadamard(n), dtype=torch.float32, device=x.device)
    y = x @ H
    if axis != -1:
        y = torch.transpose(y, -1, axis)
    return y


def ifwht(x, axis=-1):
    """Inverse Walsh–Hadamard Transform."""
    if axis != -1:
        x = torch.transpose(x, -1, axis)
    n = x.shape[-1]
    H = torch.tensor(hadamard(n), dtype=torch.float32, device=x.device)
    y = x @ H / n
    if axis != -1:
        y = torch.transpose(y, -1, axis)
    return y


# ==========================================================
# 🔹 Thresholding Layers
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
# 🔹 Hadamard Residual Block
# ==========================================================
class HadamardResidualBlock(nn.Module):
    """Spectral residual block using Hadamard transform."""
    def __init__(self, dim, hidden_dim=None, threshold_type='soft'):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim or dim
        self.v = nn.Parameter(torch.rand(dim))

        if threshold_type == 'soft':
            self.thresh = SoftThresholding(dim)
        elif threshold_type == 'hard':
            self.thresh = HardThresholding(dim)
        else:
            raise ValueError("threshold_type must be 'soft' or 'hard'")

        self.fc1 = nn.Linear(dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        residual = x
        # ---- Hadamard spectral modulation ----
        y = fwht(x)
        y = y * self.v
        y = self.thresh(y)
        y = ifwht(y)
        # ---- Linear refinement ----
        y = F.relu(self.fc1(y))
        y = self.fc2(y)
        # ---- Residual + normalization ----
        y = self.norm(y + residual)
        return F.relu(y)


# ==========================================================
# 🔹 Full Model (Auto Channel Detection)
# ==========================================================
class HadamardResidualNet(nn.Module):
    """
    Hadamard Residual Net (HT-FCB)
    Automatically detects the input channel count (C) on first forward pass.
    Works for any (B, C, H, W) input with 5x5 patches.
    """
    def __init__(self, window_size=5, num_blocks=3, hidden_dim=128, threshold_type='soft'):
        super().__init__()
        self.window_size = window_size
        self.num_blocks = num_blocks
        self.hidden_dim = hidden_dim
        self.threshold_type = threshold_type

        # Layers will be built dynamically
        self.fc_in = None
        self.blocks = None
        self._last_channels = None

        # Output head
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    # -------------------------
    def _initialize_layers(self, in_channels):
        """Initialize layers based on detected channel count."""
        in_dim = in_channels * self.window_size * self.window_size
        device = next(self.fc_out.parameters()).device  # ensures same device

        self.fc_in = nn.Linear(in_dim, self.hidden_dim).to(device)
        self.blocks = nn.Sequential(
            *[HadamardResidualBlock(self.hidden_dim, threshold_type=self.threshold_type)
              for _ in range(self.num_blocks)]
        ).to(device)

        self._last_channels = in_channels
        print(f"[Init] Detected {in_channels} channels → "
              f"fc_in({in_dim}→{self.hidden_dim}), "
              f"{self.num_blocks} Hadamard blocks ({self.threshold_type})")

    # -------------------------
    def forward(self, x):
        B, C, H, W = x.shape

        # Initialize or rebuild if channels changed
        if self.fc_in is None or self._last_channels != C:
            self._initialize_layers(C)

        ws = self.window_size
        patches = F.unfold(x, kernel_size=ws, stride=1)   # (B, C*ws*ws, N)
        patches = patches.transpose(1, 2)                 # (B, N, C*ws*ws)
        out = F.relu(self.fc_in(patches))
        out = self.blocks(out)
        out = self.fc_out(out)                            # (B, N, 1)

        # reshape to (B, 1, H-ws+1, W-ws+1)
        out_h, out_w = H - ws + 1, W - ws + 1
        out = out.transpose(1, 2).reshape(B, 1, out_h, out_w)
        return out


# ==========================================================
# 🔹 Example Usage
# ==========================================================
if __name__ == "__main__":
    torch.manual_seed(0)
    model = HadamardResidualNet(window_size=5, num_blocks=4, hidden_dim=64, threshold_type='soft')

    # Single-channel input
    dummy1 = torch.randn(8, 1, 5, 5)
    print("\nSingle-channel test:")
    out1 = model(dummy1)
    print("Output shape:", out1.shape)

    # Multi-channel input
    dummy2 = torch.randn(8, 12, 5, 5)
    print("\nMulti-channel test:")
    out2 = model(dummy2)
    print("Output shape:", out2.shape)
