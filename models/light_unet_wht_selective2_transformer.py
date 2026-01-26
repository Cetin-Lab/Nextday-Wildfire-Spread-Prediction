# -*- coding: utf-8 -*-
"""
Created on Sat Jan 10 23:35:31 2026

@author: olivi
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ============================================================
# Accurate Butterfly-Based FWHT and iFWHT
# ============================================================

def fwht_butterfly(x: torch.Tensor) -> torch.Tensor:
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

def ifwht_butterfly(x: torch.Tensor) -> torch.Tensor:
    x = fwht_butterfly(x)
    return x / x.shape[-1]

# ============================================================
# Hadamard → Walsh ordering
# ============================================================

def hadamard_to_walsh(H: torch.Tensor) -> torch.Tensor:
    sign_changes = torch.sum(H[:, :-1] * H[:, 1:] < 0, dim=1)
    return H[torch.argsort(sign_changes)]

# ============================================================
# Learnable FWHT Transform
# ============================================================

class LearnableFWHTTransform(nn.Module):
    def __init__(self, size: int, mode: str = "hadamard", learnable: bool = False, device: str = "cpu"):
        super().__init__()
        assert (size & (size - 1)) == 0, "size must be power of 2"

        I = torch.eye(size, device=device)
        T = fwht_butterfly(I)

        if mode.lower() == "walsh":
            T = hadamard_to_walsh(T)

        # If learnable=False, it's still a Parameter but frozen (as you checked earlier)
        self.T = nn.Parameter(T, requires_grad=learnable)
        self.size = size

    def forward(self, x: torch.Tensor, axis: int = -1, inverse: bool = False) -> torch.Tensor:
        T = self.T.T if not inverse else self.T
        x = x.transpose(axis, -1)
        x = x @ T
        x = x.transpose(-1, axis)
        return x / self.size if inverse else x

# ============================================================
# Threshold Layer
# ============================================================

class Thresholding(nn.Module):
    def __init__(self, t_shape, mode: str = "soft", init_scale: float = 0.1, learnable: bool = True, ste_for_hard: bool = True):
        super().__init__()
        self.mode = mode
        self.ste = ste_for_hard
        T = torch.rand(t_shape) * init_scale
        # In your design: thresholds are learnable (you pass learnable=True in WHT2D)
        self.T = nn.Parameter(T) if learnable else nn.Parameter(T, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Tabs = self.T.abs()
        ax = x.abs()

        if self.mode == "soft":
            return torch.sign(x) * F.relu(ax - Tabs)

        if self.mode == "hard":
            mask = (ax > Tabs).float()
            y = x * mask
            return y + (x - y).detach() if self.ste else y

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

    def __init__(self, height: int, width: int, mode: str, learnable_T: bool, threshold_mode: str):
        super().__init__()
        assert (height & (height - 1)) == 0, "height must be power of 2"
        assert (width  & (width  - 1)) == 0, "width must be power of 2"

        self.H = height
        self.W = width

        self.fwht_H = LearnableFWHTTransform(height, mode, learnable_T)
        self.fwht_W = LearnableFWHTTransform(width,  mode, learnable_T)

        # Always learnable
        self.v = nn.Parameter(torch.ones(height, width))

        # Threshold always learnable here
        self.threshold = Thresholding(
            (height, width),
            mode=threshold_mode,
            init_scale=0.1,
            learnable=True
        )

    def forward_transform(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fwht_W(x, axis=3)
        x = self.fwht_H(x, axis=2)
        return x

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fwht_H(x, axis=2, inverse=True)
        x = self.fwht_W(x, axis=3, inverse=True)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        assert (H, W) == (self.H, self.W), f"Expected {(self.H,self.W)}, got {(H,W)}"

        X = self.forward_transform(x)
        X = X * self.v
        X = self.threshold(X)
        Y = self.inverse_transform(X)
        return Y

# ============================================================
# Transformer pieces (bottleneck)
# ============================================================

class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim: int, heads: int = 4, dim_head: int = 32, dropout: float = 0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(
                nn.Linear(inner_dim, dim),
                nn.Dropout(dropout),
            )
            if project_out
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, C)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # (B, H, N, d)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)

class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int = 4, dim_head: int = 32, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = Attention(dim=dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.ln2 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim=dim, hidden_dim=int(dim * mlp_ratio), dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attn(self.ln1(x)) + x
        x = self.ff(self.ln2(x)) + x
        return x

# ============================================================
# UNet Components
# ============================================================

class DoubleConvWHT(nn.Module):
    """
    If use_wht=False: standard DoubleConv (3x3 -> 3x3)
    If use_wht=True : WHT2D -> Conv3x3(in_ch->out_ch) + BN + ReLU  (single conv)
    """
    def __init__(self, in_ch: int, out_ch: int, H: int, W: int, use_wht: bool,
                 transform_mode: str, learnable_T: bool, threshold_mode: str):
        super().__init__()
        self.use_wht = use_wht
        self.H, self.W = H, W
        self.relu = nn.ReLU(inplace=True)

        if use_wht:
            self.wht = WHT2D(H, W, transform_mode, learnable_T, threshold_mode)
            self.conv2 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
            self.bn2   = nn.BatchNorm2d(out_ch)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
            self.bn1   = nn.BatchNorm2d(out_ch)
            self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
            self.bn2   = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_wht:
            B, C, H, W = x.shape
            assert (H, W) == (self.H, self.W), f"Expected {(self.H,self.W)}, got {(H,W)}"

            x = x.reshape(B * C, 1, H, W)
            x = self.wht(x)
            x = x.reshape(B, C, H, W)

            x = self.relu(self.bn2(self.conv2(x)))
            return x

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, bilinear: bool = True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_ch, out_ch)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)
            self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

# ============================================================
# FULL UNET WITH WHT SELECTOR + BOTTLENECK TRANSFORMER
# ============================================================

class LiteUNetSelectiveWHT(nn.Module):
    def __init__(
        self,
        n_channels: int = 12,
        n_classes: int = 1,
        base_c: int = 4,
        bilinear: bool = True,
        use_wht: dict | None = None,
        transform_mode: str = "walsh",
        learnable_T: bool = False,
        threshold_mode: str = "soft",
        use_bottleneck_transformer: bool = True,
        trans_heads: int = 4,
        trans_dim_head: int = 32,
        trans_mlp_ratio: float = 4.0,
        trans_dropout: float = 0.0,
        trans_layers: int = 2,
    ):
        super().__init__()

        if use_wht is None:
            use_wht = {k: False for k in ["inc", "down1", "down2", "down3", "down4"]}

        c1, c2, c3, c4, c5 = base_c, base_c * 2, base_c * 4, base_c * 8, base_c * 16
        factor = 2 if bilinear else 1

        # Encoder
        self.inc = DoubleConvWHT(n_channels, c1, 64, 64, use_wht["inc"],
                                 transform_mode, learnable_T, threshold_mode)

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
            DoubleConvWHT(c4, c5 // factor, 4, 4, use_wht["down4"],
                          transform_mode, learnable_T, threshold_mode)
        )

        # Bottleneck Transformer
        self.use_bottleneck_transformer = use_bottleneck_transformer
        bottleneck_dim = c5 // factor  # channels at x5

        if use_bottleneck_transformer:
            blocks = []
            for _ in range(max(1, int(trans_layers))):
                blocks.append(
                    TransformerBlock(
                        dim=bottleneck_dim,
                        heads=trans_heads,
                        dim_head=trans_dim_head,
                        mlp_ratio=trans_mlp_ratio,
                        dropout=trans_dropout
                    )
                )
            self.bottleneck_transformer = nn.Sequential(*blocks)
        else:
            self.bottleneck_transformer = nn.Identity()

        # Decoder
        self.up1 = Up(c5 // factor + c4, c4 // factor, bilinear)
        self.up2 = Up(c4 // factor + c3, c3 // factor, bilinear)
        self.up3 = Up(c3 // factor + c2, c2 // factor, bilinear)
        self.up4 = Up(c2 // factor + c1, c1, bilinear)

        self.outc = OutConv(c1, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)      # (B,c1,64,64)
        x2 = self.down1(x1)   # (B,c2,32,32)
        x3 = self.down2(x2)   # (B,c3,16,16)
        x4 = self.down3(x3)   # (B,c4,8,8)
        x5 = self.down4(x4)   # (B,c5//factor,4,4)

        if self.use_bottleneck_transformer:
            B, C, H, W = x5.shape
            tokens = rearrange(x5, "b c h w -> b (h w) c")  # (B,16,C)
            tokens = self.bottleneck_transformer(tokens)
            x5 = rearrange(tokens, "b (h w) c -> b c h w", h=H, w=W)

        x = self.up1(x5, x4)
        x = self.up2(x,  x3)
        x = self.up3(x,  x2)
        x = self.up4(x,  x1)
        return self.outc(x)

# ============================================================
# Standalone test
# ============================================================

if __name__ == "__main__":
    use_wht = {
        "inc":   True,
        "down1": True,
        "down2": True,
        "down3": True,
        "down4": True,
    }

    print("\n===== LiteUNetSelectiveWHT + Bottleneck Transformer =====")
    for k, v in use_wht.items():
        print(f" {k}: {'WHT2D' if v else 'Conv2D'}")
    print("=========================================================\n")

    model = LiteUNetSelectiveWHT(
        n_channels=12,
        n_classes=1,
        base_c=4,
        use_wht=use_wht,
        transform_mode="walsh",
        learnable_T=False,
        threshold_mode="soft",
        use_bottleneck_transformer=True,
        trans_heads=4,
        trans_dim_head=32,
        trans_layers=2
    )

    x = torch.randn(2, 12, 64, 64)
    y = model(x)

    # Print key parameter flags
    for n, p in model.named_parameters():
        if "fwht_" in n or ".v" in n or "threshold.T" in n:
            print(n, p.requires_grad, p.shape)

    print("Input shape: ", x.shape)
    print("Output shape:", y.shape)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total params:     {total_params:,}")
    print(f"Trainable params: {trainable_params:,}\n")
