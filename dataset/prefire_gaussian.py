# -*- coding: utf-8 -*-
"""
Created on Fri May 30 16:32:51 2025

@author: Dell
"""

# prefire_gauss.py
# ──────────────────────────────────────────────────────────────
import numpy as np
import scipy.ndimage as ndi

# ── predefined σ menus ────────────────────────────────────────
SIGMA_PROFILES = {
    #   name       σ values (pixels)         description
    "fine":      [0.4, 0.8],                 # very sharp halo
    "moderate":  [0.7, 1.5, 3.0],            # balanced default
    "wide":      [1.0, 3.0, 6.0],            # generous spread
    "ultra":     [2.0, 4.0, 8.0, 12.0],      # very soft, long-range
    # add your own variants here ↓
    "log5":      np.logspace(-1, 0.7, 5),    # 0.1 … 5.0, 5 values
    "linear7":   np.linspace(0.5, 7.0, 7),   # 0.5 … 7.0, 7 values
}

# ── core routine (unchanged) ─────────────────────────────────
def gaussian_prob_map(mask: np.ndarray,
                      sigmas,
                      clip=(0.0, 1.0),
                      combine="union"):
    """
    Binary mask -> single soft probability map from multiple Gaussian blurs.
    """
    blurred = [ndi.gaussian_filter(mask.astype(np.float32), s, mode="nearest")
               for s in sigmas]
    stack = np.clip(np.stack(blurred, 0), *clip)

    if combine == "mean":
        prob = stack.mean(0)
    elif combine == "max":
        prob = stack.max(0)
    elif combine == "union":
        prob = 1.0 - np.prod(1.0 - stack, 0)
    else:
        raise ValueError("combine must be 'mean', 'max', or 'union'")
    return np.clip(prob, *clip).astype(np.float32)

# ── convenience wrapper ──────────────────────────────────────
def soften_prevfire(mask: np.ndarray,
                    profile: str = "moderate",
                    combine: str = "union"):
    """
    Convert binary PrevFireMask → probability map using a named σ profile.
    """
    if profile not in SIGMA_PROFILES:
        raise KeyError(f"Profile '{profile}' not found. "
                       f"Available: {list(SIGMA_PROFILES)}")
    return gaussian_prob_map(mask,
                             sigmas=SIGMA_PROFILES[profile],
                             combine=combine)



import matplotlib.pyplot as plt
# from prefire_gauss import soften_prevfire, SIGMA_PROFILES
H, W = 64, 64
yy, xx = np.mgrid[:H, :W]
binary_mask = ((yy - H//2)**2 + (xx - W//2)**2 <= 15**2).astype(np.uint8)
prob_map = soften_prevfire(binary_mask, profile="wide")  # 1.0,3.0,6.0

# Visualise all profiles on a toy mask
toy = binary_mask
fig, axs = plt.subplots(1, len(SIGMA_PROFILES)+1, figsize=(3*(len(SIGMA_PROFILES)+1),3))
axs[0].imshow(toy); axs[0].set_title("binary"); axs[0].axis("off")

for i,(name,sigmas) in enumerate(SIGMA_PROFILES.items(), start=1):
    axs[i].imshow(soften_prevfire(toy, profile=name))
    axs[i].set_title(f"{name}\nσ={list(np.round(sigmas,2))}"); axs[i].axis("off")
plt.tight_layout(); plt.show()
