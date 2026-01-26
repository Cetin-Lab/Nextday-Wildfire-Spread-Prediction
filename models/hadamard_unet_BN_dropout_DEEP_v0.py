# -*- coding: utf-8 -*-
"""
Created on Mon May 19 15:41:28 2025

@author: Dell
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.linalg import hadamard
from torchsummary import summary

# Hadamard transforms
def fwht(x, axis=-1):
    if axis != -1:
        x = torch.transpose(x, -1, axis)
    n = x.shape[-1]
    assert (n & (n - 1) == 0), "Input size must be power of 2"
    H = torch.tensor(hadamard(n), dtype=torch.float32, device=x.device)
    y = x @ H
    if axis != -1:
        y = torch.transpose(y, -1, axis)
    return y

def ifwht(x, axis=-1):
    if axis != -1:
        x = torch.transpose(x, -1, axis)
    n = x.shape[-1]
    H = torch.tensor(hadamard(n), dtype=torch.float32, device=x.device)
    y = x @ H / n
    if axis != -1:
        y = torch.transpose(y, -1, axis)
    return y

# Per-channel Thresholding: T shape should be (1, C, H, W)
class Thresholding(nn.Module):
    """
    Thresholding layer with switchable 'soft' or 'hard' modes.

    Args:
        t_shape: tuple, shape of the threshold tensor T (broadcastable to x).
                 e.g., (1, C, H, W) for per-channel spatial thresholds,
                       (1, C, 1, 1) for per-channel only, or (1,1,1,1) global.
        mode: 'soft' or 'hard'
        init_scale: thresholds initialized ~ U(0, init_scale)
        learnable: if False, T is frozen (no grad)
        ste_for_hard: if True, uses straight-through estimator in 'hard' mode
                      (passes gradient as identity through masked zeros)
    """
    def __init__(self,
                 t_shape,
                 mode: str = "soft",
                 init_scale: float = 0.1,
                 learnable: bool = True,
                 ste_for_hard: bool = True):
        super().__init__()
        self.mode = mode.lower()
        self.ste_for_hard = ste_for_hard

        T = torch.rand(t_shape) * init_scale
        self.T = nn.Parameter(T) if learnable else nn.Parameter(T, requires_grad=False)

    def set_mode(self, mode: str):
        self.mode = mode.lower()

    def forward(self, x):
        Tabs = self.T.abs()
        ax = x.abs()

        if self.mode == "soft":
            # Soft thresholding: sign(x) * relu(|x| - T)
            y = torch.sign(x) * F.relu(ax - Tabs)

        elif self.mode == "hard":
            # Hard thresholding: x * 1{|x| > T}
            mask = (ax > Tabs).to(x.dtype)
            y = x * mask
            if self.ste_for_hard:
                # Straight-through estimator for gradients:
                # forward: y; backward: dy/dx ≈ 1 everywhere
                y = y + (x - y).detach()
        else:
            raise ValueError("mode must be 'soft' or 'hard'")

        return y

class HadamardUnet(nn.Module):
    def __init__(self, input_channels=12, input_size=64, output_channels=1, dropout_rate=0.3):
        super().__init__()
        self.height = input_size // 2
        self.width  = input_size // 2
        C = 4  # channels in all HT blocks below

        self.conv1 = nn.Conv2d(input_channels, C, kernel_size=4, stride=2, padding=1)
        self.bn1   = nn.BatchNorm2d(C)
        self.dropout1 = nn.Dropout2d(dropout_rate)

        self.conv2 = nn.Conv2d(C, C, kernel_size=7, stride=1, padding=3)
        self.bn2   = nn.BatchNorm2d(C)
        self.dropout2 = nn.Dropout2d(dropout_rate)

        self.conv3 = nn.Conv2d(C, C, kernel_size=7, stride=1, padding=3)
        self.bn3   = nn.BatchNorm2d(C)
        self.dropout3 = nn.Dropout2d(dropout_rate)

        self.deconv1 = nn.ConvTranspose2d(C, C, kernel_size=7, stride=1, padding=3)
        self.bn4     = nn.BatchNorm2d(C)
        self.dropout4 = nn.Dropout2d(dropout_rate)

        self.deconv2 = nn.ConvTranspose2d(C, C, kernel_size=7, stride=1, padding=3)
        self.bn5     = nn.BatchNorm2d(C)
        self.dropout5 = nn.Dropout2d(dropout_rate)

        self.deconv3 = nn.ConvTranspose2d(C, output_channels, kernel_size=4, stride=2, padding=1)

        # --- Per-channel v masks: (1, C, H, W)
        self.v1 = nn.Parameter(torch.rand(1, C, self.height, self.width))
        self.v2 = nn.Parameter(torch.rand(1, C, self.height, self.width))
        self.v3 = nn.Parameter(torch.rand(1, C, self.height, self.width))
        self.v4 = nn.Parameter(torch.rand(1, C, self.height, self.width))
        self.v5 = nn.Parameter(torch.rand(1, C, self.height, self.width))

        # --- Per-channel thresholds T: (1, C, H, W)
        self.ST1 = Thresholding((1, C, self.height, self.width))
        self.ST2 = Thresholding((1, C, self.height, self.width))
        self.ST3 = Thresholding((1, C, self.height, self.width))
        self.ST4 = Thresholding((1, C, self.height, self.width))
        self.ST5 = Thresholding((1, C, self.height, self.width))

    def forward(self, x):
        # Block 1
        x1 = F.relu(self.bn1(self.conv1(x)))
        x1 = self.dropout1(x1)

        x2 = fwht(fwht(x1, axis=-1), axis=-2)
        x3 = self.v1 * x2                 # (1,C,H,W) * (B,C,H,W)
        x4 = self.ST1(x3)                 # T: (1,C,H,W)
        x5 = ifwht(ifwht(x4, axis=-1), axis=-2)

        # Block 2
        x6 = F.relu(self.bn2(self.conv2(x5)))
        x6 = self.dropout2(x6)

        x7  = fwht(fwht(x6, axis=-1), axis=-2)
        x8  = self.v2 * x7
        x9  = self.ST2(x8)
        x10 = ifwht(ifwht(x9, axis=-1), axis=-2)

        # Block 3
        x11 = F.relu(self.bn3(self.conv3(x10)))
        x11 = self.dropout3(x11)

        x12 = fwht(fwht(x11, axis=-1), axis=-2)
        x13 = self.v3 * x12
        x14 = self.ST3(x13)
        x15 = ifwht(ifwht(x14, axis=-1), axis=-2)

        # Block 4
        x16 = F.relu(self.bn4(self.deconv1(x15)))
        x16 = self.dropout4(x16)
        x16 = x16 + x11                   # skip (feature space)

        x17 = fwht(fwht(x16, axis=-1), axis=-2)
        x18 = self.v4 * x17
        x19 = self.ST4(x18) + x9          # residual in HT domain
        x20 = ifwht(ifwht(x19, axis=-1), axis=-2)

        # Block 5
        x21 = F.relu(self.bn5(self.deconv2(x20)))
        x21 = self.dropout5(x21)
        x21 = x21 + x6                    # skip (feature space)

        x22 = fwht(fwht(x21, axis=-1), axis=-2)
        x23 = self.v5 * x22
        x24 = self.ST5(x23) + x4          # residual in HT domain
        x25 = ifwht(ifwht(x24, axis=-1), axis=-2)

        # Output
        x_out = self.deconv3(x25)
        return x_out

# ────────────── Stand‑alone test ───────────────────────────────────


# # pip install torchsummary
# from torchsummary import summary
# import torch

# device = "cuda" if torch.cuda.is_available() else "cpu"

# model = HadamardUnet(input_channels=12, input_size=64, output_channels=1).eval().to(device)

# summary(model, input_size=(12, 64, 64), device=device)  # excludes batch size
# print("— torchsummary done —", flush=True)

# dummy = torch.randn(2, 12, 64, 64, device=device)
# with torch.no_grad():
#     out = model(dummy)
# print("Output shape:", out.shape, flush=True)
# Create model
import torch
from torchsummary import summary

# ────────────────────────────────
# Helper: Count parameters
# ────────────────────────────────
def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🔢 Total Parameters: {total_params:,}")
    print(f"🧠 Trainable Parameters: {trainable_params:,}")

# ────────────────────────────────
# Helper: Count total layers
# ────────────────────────────────
def count_layers(model):
    layer_count = sum(1 for _ in model.modules() if not isinstance(_, torch.nn.Sequential) and not isinstance(_, torch.nn.ModuleList) and not isinstance(_, torch.nn.ModuleDict) and type(_) != type(model))
    print(f"🧱 Total Layers: {layer_count}")

# ────────────────────────────────
# Helper: Measure inference time
# ────────────────────────────────
import time
def measure_inference_time(model, device, input_shape=(2, 12, 64, 64), n_runs=50):
    model.eval()
    dummy_input = torch.randn(*input_shape).to(device)
    # Warm-up
    with torch.no_grad():
        for _ in range(5):
            _ = model(dummy_input)
    # Timing
    start = time.time()
    with torch.no_grad():
        for _ in range(n_runs):
            _ = model(dummy_input)
    end = time.time()
    avg_time = (end - start) / n_runs
    print(f"⏱️ Average Inference Time over {n_runs} runs: {avg_time:.6f} sec")

device = "cuda" if torch.cuda.is_available() else "cpu"
# ────────────────────────────────
# Main: Load model
# ────────────────────────────────
model = HadamardUnet(input_channels=40, input_size=128, output_channels=1).to(device)

# Count total/trainable parameters
count_parameters(model)

# Count total layers
count_layers(model)

# Show model summary
print("\n📋 Model Summary:")
summary(model, input_size=(40, 128, 128), device=str(device))  # excludes batch size

# Inference time benchmark
measure_inference_time(model, device, input_shape=(2, 40, 128, 128))

# Sanity check output shape
dummy = torch.randn(2, 40, 128, 128, device=device)
with torch.no_grad():
    out = model(dummy)
print(f"✅ Output shape (batch=2): {out.shape}", flush=True)
