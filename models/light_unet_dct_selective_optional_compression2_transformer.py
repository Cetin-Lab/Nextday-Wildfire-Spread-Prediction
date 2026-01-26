# -*- coding: utf-8 -*-
"""
LiteUNetSelectiveDCT (OPTIONAL COMPRESSION) + Bottleneck Transformer

- Replaces WHT2D with a 2D DCT-II block (orthonormal).
- Optional low-frequency compression via coefficient cropping.
- DCT-mode block: DCT2D -> Conv3x3(in_ch->out_ch) + BN + ReLU  (single conv for mapping)
- Optional Transformer inserted at bottleneck (after down4, before up1)

@author: olivi (adapted)
"""

import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

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

    n = torch.arange(N, device=device, dtype=dtype).view(1, N)
    k = torch.arange(N, device=device, dtype=dtype).view(N, 1)

    C = torch.cos(torch.pi * (n + 0.5) * k / N)
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

        # v and thresholds are learnable (and remain so even with compression)
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
    def __init__(self, dim: int, heads: int = 4, dim_head: int = 32,
                 mlp_ratio: float = 4.0, dropout: float = 0.0):
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
# DoubleConv with optional DCT
# ============================================================

class DoubleConvDCT(nn.Module):
    """
    If use_dct=False: standard DoubleConv (3x3 -> 3x3)
    If use_dct=True : DCT2D -> Conv3x3(in_ch->out_ch) + BN + ReLU  (single conv)
    """
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

            # conv2 for channel mapping: in_ch -> out_ch
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

            x = x.reshape(B * C, 1, H, W)
            x = self.dct(x)
            x = x.reshape(B, C, H, W)

            x = self.relu(self.bn2(self.conv2(x)))
            return x

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
# FULL UNET WITH DCT SELECTOR + OPTIONAL COMPRESSION + TRANSFORMER
# ============================================================

class LiteUNetSelectiveDCT(nn.Module):
    def __init__(
        self,
        n_channels=12,
        n_classes=1,
        base_c=4,
        bilinear=True,
        use_dct=None,
        compression_ratio=1.0,     # 1.0 disables compression (default)
        learnable_T=False,
        threshold_mode="soft",
        # ---- Bottleneck transformer args ----
        use_bottleneck_transformer=True,
        trans_heads=4,
        trans_dim_head=32,
        trans_mlp_ratio=4.0,
        trans_dropout=0.0,
        trans_layers=2
    ):
        super().__init__()

        if use_dct is None:
            use_dct = {k: False for k in ["inc","down1","down2","down3","down4"]}

        c1, c2, c3, c4, c5 = base_c, base_c*2, base_c*4, base_c*8, base_c*16
        factor = 2 if bilinear else 1

        # Encoder
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

        # Bottleneck Transformer
        self.use_bottleneck_transformer = use_bottleneck_transformer
        bottleneck_dim = c5 // factor  # channels at x5 (4x4)

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

        # Decoder (unchanged)
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
        x5 = self.down4(x4)  # (B, Cb, 4, 4)

        if self.use_bottleneck_transformer:
            B, C, H, W = x5.shape
            tokens = rearrange(x5, "b c h w -> b (h w) c")  # (B, 16, C)
            tokens = self.bottleneck_transformer(tokens)
            x5 = rearrange(tokens, "b (h w) c -> b c h w", h=H, w=W)

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

    print("\n===== LiteUNetSelectiveDCT + Bottleneck Transformer =====")
    for k, v in use_dct.items():
        print(f" {k}: {'DCT2D' if v else 'Conv2D'}")
    print("=========================================================\n")

    model = LiteUNetSelectiveDCT(
        n_channels=12,
        n_classes=1,
        base_c=4,
        use_dct=use_dct,
        compression_ratio=1,   # set 1.0 to disable compression
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
        if ".v" in n or "threshold.T" in n or "C_H" in n or "C_W" in n:
            print(n, p.requires_grad, p.shape)

    print("Input shape:  ", x.shape)
    print("Output shape: ", y.shape)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total params:     {total_params:,}")
    print(f"Trainable params: {trainable_params:,}\n")
