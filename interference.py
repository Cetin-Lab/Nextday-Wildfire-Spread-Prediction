# -*- coding: utf-8 -*-
"""
Created on Thu Jan 22 15:08:12 2026

Inference script for Deep HT-UNet (HadamardUnet) with Gaussian preprocessing.
Iterates over the ENTIRE test dataset and saves per-sample visualizations:

Panel 1: Pre-fire mask
Panel 2: Ground truth
Panel 3: Prediction with mismatch overlays:
  - False Positives (pred=1, gt=0): yellow
  - False Negatives (pred=0, gt=1): blue
"""

import os, sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

# ── Paths / Imports ───────────────────────────────────────────────
sys.path += [
    "D:/wildfire/wildfire_detection/codetfAE1",
    "D:/wildfire/wildfire_detection/codetfAE1/models",
]

from hadamard_unet_BN_dropout_DEEP_v0 import HadamardUnet
from dataset_adjust_pre_post_gaussian_B import make_dataset, ModeKeys
from constants import INPUT_FEATURES

# ── Config ───────────────────────────────────────────────────────
MODEL_PATH = r"D:\wildfire\wildfire_detection\codetfAE1\gaussian_sweep_results_DEEP_v0\fine_01\best_model.pth"
OUT_DIR    = r"D:\wildfire\paper\interference2"
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
THRESH = 0.5

# Save ALL samples if None; otherwise cap to N (e.g., 200)
MAX_SAVE = None


# ── Hyperparameters (match training) ─────────────────────────────
class HParams:
    train_path = "D:/wildfire/wildfire_detection/archive/next_day_wildfire_spread_train_*.tfrecord"
    eval_path  = "D:/wildfire/wildfire_detection/archive/next_day_wildfire_spread_eval_*.tfrecord"
    test_path  = "D:/wildfire/wildfire_detection/archive/next_day_wildfire_spread_test_*.tfrecord"

    input_features  = list(INPUT_FEATURES)
    output_features = ["FireMask"]
    data_sample_size = 64
    sample_size      = 64
    output_sample_size = 64
    input_sequence_length = 1
    output_sequence_length = 1
    azimuth_in_channel  = "th"
    azimuth_out_channel = None

    shuffle_buffer_size = 500
    compression_type   = ""
    random_flip  = True
    random_rotate = False
    random_crop   = False
    downsample_threshold = 0.3
    binarize_output = True

    batch_size     = 32
    epochs         = 100
    steps_per_epoch= 1000
    learning_rate  = 1e-4
    pos_weight     = 3.0
    run_threshold_optimization = False

    use_prefire_noise    = True
    use_prefire_gaussian = True
    combine_fire_masks   = True

    gaussian_profile = "moderate"
    gaussian_combine = "mean"

hp = HParams()


# ── Visualization helper (USE YOUR EXACT FP/FN COLORS) ───────────
def plot_example(pre_mask, gt, pred, idx, save_dir,
                 title_size=16, title_weight="bold",
                 dpi=300, show=True):
    """
    Saves (and optionally shows) a single 3-panel plot:
      - Pre-fire mask (gray + orange)
      - Ground truth (gray + orange)
      - Prediction (gray + orange, FP=yellow, FN=blue)
    """
    base_cmap = colors.ListedColormap(["silver", "orangered"])
    base_norm = colors.BoundaryNorm([0, 0.5, 1], base_cmap.N)

    # Ensure ints for comparisons and display
    pre_mask = pre_mask.astype(np.int32)
    gt = gt.astype(np.int32)
    pred = pred.astype(np.int32)

    # Overlays for mismatches
    fp = ((pred == 1) & (gt == 0)).astype(np.int32)  # False positives
    fn = ((pred == 0) & (gt == 1)).astype(np.int32)  # False negatives

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # --- Pre-fire mask ---
    axes[0].imshow(pre_mask, cmap=base_cmap, norm=base_norm, interpolation="nearest")
    axes[0].axis("off")
    axes[0].set_title("Pre-fire", fontsize=title_size, fontweight=title_weight)

    # --- Ground truth ---
    axes[1].imshow(gt, cmap=base_cmap, norm=base_norm, interpolation="nearest")
    axes[1].axis("off")
    axes[1].set_title("Ground Truth", fontsize=title_size, fontweight=title_weight)

    # --- Prediction ---
    axes[2].imshow(pred, cmap=base_cmap, norm=base_norm, interpolation="nearest")

    # Overlay false positives in yellow (only where fp==1)
    axes[2].imshow(
        fp,
        cmap=colors.ListedColormap([(0, 0, 0, 0), (1, 1, 0, 1)]),
        alpha=0.7,
        interpolation="nearest"
    )

    # Overlay false negatives in blue (only where fn==1)
    axes[2].imshow(
        fn,
        cmap=colors.ListedColormap([(0, 0, 0, 0), (0, 0, 1, 1)]),
        alpha=0.7,
        interpolation="nearest"
    )

    axes[2].axis("off")
    axes[2].set_title("Prediction", fontsize=title_size, fontweight=title_weight)

    # Save
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"example_{idx:05d}.png")
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    print("🖼️ Saved", save_path)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return save_path, int(fp.sum()), int(fn.sum())


def main():
    # ── Dataset ──────────────────────────────────────────────────
    test_ds = make_dataset(hp, mode=ModeKeys.PREDICT)

    # ── Model ────────────────────────────────────────────────────
    model = HadamardUnet(len(hp.input_features), hp.sample_size, 1).to(DEVICE)
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    print("✅ Loaded model from", MODEL_PATH)
    print("✅ Saving to:", OUT_DIR)
    print("✅ Device:", DEVICE)

    global_idx = 0
    total_fp = 0
    total_fn = 0

    # Iterate over ALL batches in the TF dataset
    for feat_batch, lab_batch in test_ds:
        feat_np = feat_batch.numpy()  # (B, H, W, C)
        lab_np  = lab_batch.numpy()   # (B, H, W, 1)

        x = torch.from_numpy(feat_np).permute(0, 3, 1, 2).float().to(DEVICE)

        with torch.no_grad():
            prob = torch.sigmoid(model(x)).squeeze(1).cpu().numpy()  # (B,H,W)

        gt = lab_np[..., 0].astype(np.int32)  # (B,H,W)

        # Pre-fire mask: last input channel assumed to be PrevFireMask-like signal
        pre = (feat_np[:, :, :, -1] > THRESH).astype(np.int32)

        pred_bin = (prob > THRESH).astype(np.int32)

        B = pred_bin.shape[0]
        for j in range(B):
            if (MAX_SAVE is not None) and (global_idx >= MAX_SAVE):
                break

            save_path, fp_cnt, fn_cnt = plot_example(
                pre[j], gt[j], pred_bin[j],
                global_idx, OUT_DIR,
                show=False  # change to True if you want pop-up windows
            )

            total_fp += fp_cnt
            total_fn += fn_cnt
            print(f"   FP={fp_cnt}  FN={fn_cnt}")

            global_idx += 1

        if (MAX_SAVE is not None) and (global_idx >= MAX_SAVE):
            break

    print(f"\n✅ Done. Saved {global_idx} examples.")
    print(f"Total FP pixels (over saved examples): {total_fp}")
    print(f"Total FN pixels (over saved examples): {total_fn}")


if __name__ == "__main__":
    main()
