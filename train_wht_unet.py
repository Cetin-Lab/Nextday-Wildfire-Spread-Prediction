# -*- coding: utf-8 -*-
"""
Updated on Dec 8, 2025
Integrated LiteUNetSelectiveWHT for WHT ablation experiments
"""

import os, sys, re, json, warnings
sys.path += [
    "/home/omid/Shuaiang/wildfire_detection/codetfAE1",
    "/home/omid/Shuaiang/wildfire_detection/codetfAE1/models"
]

os.environ.update({
    "OMP_NUM_THREADS": "1",
    "TF_NUM_INTRAOP_THREADS": "1",
    "TF_NUM_INTEROP_THREADS": "1",
    "TF_CPP_MIN_LOG_LEVEL": "3",
})

import torch
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt, colors

from torchinfo import summary
import tensorflow as tf
from tensorflow.keras.utils import Progbar
import logging
tf.get_logger().handlers.clear()
logging.getLogger('tensorflow').handlers.clear()
warnings.filterwarnings("ignore")

# >>> NEW: import your selective-WHT UNet <<<
from light_unet_wht_selective2_transformer import LiteUNetSelectiveWHT

from dataset_adjust_pre_post_gaussian_B import make_dataset, ModeKeys
from constants import INPUT_FEATURES
import prefire_gaussian
from losses1 import combo_loss
from metrics import (
    AUCWithMaskedClass, PrecisionWithMaskedClass, RecallWithMaskedClass,
    masked_iou, compute_best_threshold, compute_pr_auc
)

device = "cuda" if torch.cuda.is_available() else "cpu"

# ================================================================
# HYPERPARAMETERS
# ================================================================
class HParams:
    train_path = "/home/omid/Shuaiang/wildfire_detection/archive/next_day_wildfire_spread_train_*.tfrecord"
    eval_path  = "/home/omid/Shuaiang/wildfire_detection/archive/next_day_wildfire_spread_eval_*.tfrecord"
    test_path  = "/home/omid/Shuaiang/wildfire_detection/archive/next_day_wildfire_spread_test_*.tfrecord"
    input_features  = list(INPUT_FEATURES)
    output_features = ["FireMask"]
    data_sample_size = 64
    sample_size = 64
    output_sample_size = 64
    input_sequence_length = 1
    output_sequence_length = 1
    azimuth_in_channel  = "th"
    azimuth_out_channel = None
    shuffle_buffer_size = 500
    compression_type = ""
    random_flip = True
    random_rotate = False
    random_crop = False
    downsample_threshold = 0.3
    binarize_output = True
    batch_size = 32
    epochs = 500
    steps_per_epoch = 1000
    learning_rate = 1e-4
    pos_weight = 5.0
    run_threshold_optimization = False
    gaussian_profile = "moderate"
    gaussian_combine = "union"
    early_stopping_patience = 40
    

    # >>> NEW hyperparameters for WHT UNet <<<
    # >>> WHT / Transform hyperparameters <<<
    use_wht = {
        "inc": True,
        "down1": True,
        "down2": True,
        "down3": True,
        "down4": True,
    }

    transform_mode = "walsh"          # "hadamard" or "walsh"
    threshold_mode = "soft"           # "soft" or "hard"
    learnable_transform = False       # learnable Hadamard/Walsh matrix?

    # >>> Bottleneck Transformer hyperparameters <<<
    use_bottleneck_transformer = True
    trans_heads = 4
    trans_dim_head = 32
    trans_mlp_ratio = 4.0
    trans_dropout = 0.0
    trans_layers = 2

# ------------------------------------------------------------------
# 3.  Utility helpers
# ------------------------------------------------------------------
def get_numbered_save_dir(root:str, prefix:str) -> str:
    os.makedirs(root, exist_ok=True)
    existing = [d for d in os.listdir(root)
                if os.path.isdir(os.path.join(root, d)) and re.match(f"{prefix}\\d+", d)]
    ids = [int(re.search(r"\d+", d).group()) for d in existing] if existing else []
    name = f"{prefix}{max(ids)+1:02d}" if ids else f"{prefix}01"
    path = os.path.join(root, name)
    os.makedirs(path)
    return path

def plot_train_val_losses(train_losses, val_losses, save_path=None):
    plt.figure(figsize=(6,3))
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Val")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.grid(True); plt.legend()
    plt.tight_layout()
    if save_path: plt.savefig(save_path); print("📉  Saved " + save_path)
    plt.close()

def show_inference(n_rows, feats_tf, labs_tf, model, thr, save_path):
    seg_cmap = colors.ListedColormap(["black", "silver", "orangered"])
    seg_norm = colors.BoundaryNorm([-1, -.1, .001, 1], seg_cmap.N)
    x = torch.from_numpy(feats_tf.numpy()).permute(0,3,1,2).float().to(device)
    with torch.no_grad():
        preds = (torch.sigmoid(model(x)).squeeze(1).cpu().numpy() > thr)
    gts  = labs_tf.numpy()[...,0]
    prev = (feats_tf.numpy()[:,:,:,-1] > .5).astype(np.int32)

    plt.figure(figsize=(12,4*n_rows))
    for i in range(n_rows):
        for j, img in enumerate([prev[i], gts[i], preds[i]]):
            plt.subplot(n_rows, 3, 3*i + j + 1)
            plt.imshow(img, cmap=seg_cmap, norm=seg_norm)
            plt.axis("off")
            plt.title(["Prev", "GT", "Pred"][j])

    plt.tight_layout()
    plt.savefig(save_path)
    print("🖼️  Saved " + save_path)
    plt.close()


def evaluate_model(model, dataset, hp, threshold=0.5):
    """
    Evaluate model using masked metrics from the metrics file.
    Returns:
        iou, roc_auc, precision, recall, pr_auc, f1
    """
    auc_metric  = AUCWithMaskedClass()
    prec_metric = PrecisionWithMaskedClass()
    rec_metric  = RecallWithMaskedClass()
    ious = []

    # For PR-AUC
    all_y_true = []
    all_y_pred = []

    model.eval()
    with torch.no_grad():
        progbar = Progbar(hp.steps_per_epoch, unit_name="batch")

        for step, (feats_tf, labs_tf) in enumerate(dataset.take(hp.steps_per_epoch)):
            # ---- Convert to torch tensors ----
            x = torch.from_numpy(feats_tf.numpy()).permute(0,3,1,2).float().to(device)
            y_pred = torch.sigmoid(model(x)).squeeze(1).cpu().numpy()  # (B,H,W)
            y_true = labs_tf.numpy()[..., 0]                           # (B,H,W)

            # ---- Masked ROC-AUC ----
            auc_metric.update_state(y_true, y_pred)

            # ---- Binarize & update precision/recall ----
            bin_pred = (y_pred > threshold).astype(np.float32)
            prec_metric.update_state(y_true, bin_pred)
            rec_metric.update_state(y_true, bin_pred)

            # ---- IoU per sample ----
            for j in range(len(y_true)):
                ious.append(masked_iou(y_true[j], bin_pred[j]))

            # ---- For PR-AUC ----
            mask = (y_true != -1)
            all_y_true.append(y_true[mask])
            all_y_pred.append(y_pred[mask])

            progbar.update(step + 1)

    # ---- Combine for PR-AUC ----
    all_y_true = np.concatenate(all_y_true)
    all_y_pred = np.concatenate(all_y_pred)
    pr_auc = compute_pr_auc(all_y_true, all_y_pred)  # ← fixed (no stray '6')

    # ---- Aggregate metrics ----
    mean_iou = float(np.mean(ious))
    roc_auc  = float(auc_metric.result().numpy())
    precision = float(prec_metric.result().numpy())
    recall = float(rec_metric.result().numpy())
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return mean_iou, roc_auc, precision, recall, pr_auc, f1
    
# ------------------------------------------------------------------
# 3.5.  Pretrained initialization options
# ------------------------------------------------------------------
USE_PREVIOUS_CHECKPOINT = True          # <<< NEW: turn on/off warm start
INIT_CKPT_METRIC = "best_pr_auc"        # <<< NEW: "best_pr_auc" | "best_iou" | "best_f1"


def get_latest_experiment_dir(root: str, prefix: str):
    """
    Find the latest existing experiment directory with a given prefix,
    e.g., prefix='fine_union_' might return 'fine_union_03'.
    Returns the full path or None if none exist.
    """
    if not os.path.exists(root):
        return None

    existing = [
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d)) and re.match(f"{prefix}\\d+", d)
    ]
    if not existing:
        return None

    ids = [int(re.search(r"\d+", d).group()) for d in existing]
    latest_name = f"{prefix}{max(ids):02d}"
    return os.path.join(root, latest_name)


# ------------------------------------------------------------------
# 4.  Main sweep loop  (MINIMALLY MODIFIED)
# ------------------------------------------------------------------
import json, os, io, pandas as pd, torch
from contextlib import redirect_stdout
from torchinfo import summary


EXPERIMENT_ROOT = "gaussian_sweep_results_Full_WHT_Unet"
ALL_PROFILES = list(prefire_gaussian.SIGMA_PROFILES.keys())
COMBINE_MODES = ["mean", "max", "union"]

summary_csv = os.path.join(EXPERIMENT_ROOT, "sweep_summary.csv")
os.makedirs(EXPERIMENT_ROOT, exist_ok=True)
if not os.path.exists(summary_csv):
    with open(summary_csv, "w") as f:
        f.write(
            "profile,combine_mode,selected_by,checkpoint,"
            "test_auc,test_pr_auc,test_prec,test_rec,test_iou,test_f1\n"
        )

for profile in ["fine"]:
    for combine_mode in COMBINE_MODES:
        print(f"\n🚀  Running experiment for sigma-profile={profile} | combine={combine_mode}")
        print("-" * 60)

        hp = HParams()
        hp.gaussian_profile = profile
        hp.combine_mode = combine_mode

        prefix = f"{profile}_{combine_mode}_"
        prev_dir = get_latest_experiment_dir(EXPERIMENT_ROOT, prefix) if USE_PREVIOUS_CHECKPOINT else None

        init_ckpt_path = None
        if prev_dir is not None:
            metric_to_ckpt = {
                "best_pr_auc": "best_model_pr_auc.pth",
                "best_iou":    "best_model_iou.pth",
                "best_f1":     "best_model_f1.pth",
            }
            ckpt_name = metric_to_ckpt.get(INIT_CKPT_METRIC, "best_model_pr_auc.pth")
            candidate = os.path.join(prev_dir, ckpt_name)
            if os.path.exists(candidate):
                init_ckpt_path = candidate
                print(f"🔁 Will initialize from previous checkpoint: {init_ckpt_path}")

        save_dir = get_numbered_save_dir(EXPERIMENT_ROOT, prefix)
        os.makedirs(save_dir, exist_ok=True)
        print("📂  Saving artefacts to: " + save_dir)

        # -----------------------
        # Datasets
        # -----------------------
        train_ds = make_dataset(hp, mode=ModeKeys.TRAIN)
        val_ds   = make_dataset(hp, mode=ModeKeys.EVAL)
        test_ds  = make_dataset(hp, mode=ModeKeys.PREDICT)

        # ---------------------------------------------------------
        # >>> MINIMAL CHANGE: Replace old model with LiteUNetSelectiveWHT
        # ---------------------------------------------------------
        print("\n===== WHT Ablation Configuration =====")
        for k, v in hp.use_wht.items():
            print(f"  {k}: {'WHT2D' if v else 'Conv2D'}")
        print("======================================\n")
        
        model = LiteUNetSelectiveWHT(
            n_channels=len(hp.input_features),
            n_classes=1,
            base_c=4,
        
            # WHT config
            use_wht=hp.use_wht,
            transform_mode=hp.transform_mode,
            threshold_mode=hp.threshold_mode,
            learnable_T=hp.learnable_transform,
        
            # Bottleneck transformer config
            use_bottleneck_transformer=hp.use_bottleneck_transformer,
            trans_heads=hp.trans_heads,
            trans_dim_head=hp.trans_dim_head,
            trans_mlp_ratio=hp.trans_mlp_ratio,
            trans_dropout=hp.trans_dropout,
            trans_layers=hp.trans_layers,
        ).to(device)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total params: {total_params:,} | Trainable: {trainable_params:,}")
        print(f"Transformer enabled: {hp.use_bottleneck_transformer} | layers={hp.trans_layers}")

        if init_ckpt_path is not None:
            print(f"   🔄 Loading initial weights from: {init_ckpt_path}")
            state_dict = torch.load(init_ckpt_path)
            model.load_state_dict(state_dict, strict=False)
        else:
            print("   🚀 Training from scratch (no warm start).")

        optimiser = torch.optim.Adam(model.parameters(), lr=hp.learning_rate)

        # ---------------------------------------------------------
        # Save Model Summary and Config (unchanged)
        # ---------------------------------------------------------
        summary_path = os.path.join(save_dir, "model_summary.txt")
        with open(summary_path, "w") as f:
            with redirect_stdout(f):
                print(f"LiteUNetSelectiveWHT | sigma-profile={profile} | combine={combine_mode}")
                print(summary(
                    model,
                    input_size=(1, len(hp.input_features), hp.sample_size, hp.sample_size),
                    device=device,
                    col_names=("input_size", "output_size", "num_params", "kernel_size"),
                    depth=4
                ))
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                print(f"\nTotal params: {total_params:,}")
                print(f"Trainable params: {trainable_params:,}")

        # >>> ALSO store WHT config
        config_path = os.path.join(save_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(vars(hp), f, indent=2)

        # ---------------------------------------------------------
        # REST OF TRAINING LOOP IS UNCHANGED
        # ---------------------------------------------------------
        best_pr_auc = best_iou = best_f1 = 0.0
        best_metrics_pr = best_metrics_iou = best_metrics_f1 = None
        train_losses, val_losses, metrics_history = [], [], []
        no_improve_epochs = 0

        for epoch in range(1, hp.epochs + 1):
            print(f"\n[{profile}-{combine_mode}] Epoch {epoch}/{hp.epochs}")

            model.train()
            tloss, n = 0.0, 0
            progbar = Progbar(hp.steps_per_epoch, unit_name="batch")

            for step, (feats_tf, labs_tf) in enumerate(train_ds.take(hp.steps_per_epoch)):
                x = torch.from_numpy(feats_tf.numpy()).permute(0,3,1,2).float().to(device)
                y = torch.from_numpy(labs_tf.numpy()).permute(0,3,1,2).float().to(device)
                optimiser.zero_grad()
                loss = combo_loss(model(x), y, pos_weight=hp.pos_weight)
                loss.backward()
                optimiser.step()
                tloss += loss.item()
                n += 1
                progbar.update(step+1, values=[("loss", tloss/n)])
            train_losses.append(tloss / n)

            model.eval()
            vloss, m = 0.0, 0
            progbar = Progbar(hp.steps_per_epoch)
            with torch.no_grad():
                for step, (feats_tf, labs_tf) in enumerate(val_ds.take(hp.steps_per_epoch)):
                    x = torch.from_numpy(feats_tf.numpy()).permute(0,3,1,2).float().to(device)
                    y = torch.from_numpy(labs_tf.numpy()).permute(0,3,1,2).float().to(device)
                    vloss += combo_loss(model(x), y, pos_weight=hp.pos_weight).item()
                    m += 1
                    progbar.update(step+1, values=[("val_loss", vloss/m)])
            val_losses.append(vloss / m)

            # metrics unchanged...
            iou, auc, prec, rec, pr_auc, f1 = evaluate_model(model, val_ds, hp)

            improved = False
            if pr_auc > best_pr_auc:
                torch.save(model.state_dict(), os.path.join(save_dir, "best_model_pr_auc.pth"))
                best_pr_auc = pr_auc
                improved = True

            if iou > best_iou:
                torch.save(model.state_dict(), os.path.join(save_dir, "best_model_iou.pth"))
                best_iou = iou
                improved = True

            if f1 > best_f1:
                torch.save(model.state_dict(), os.path.join(save_dir, "best_model_f1.pth"))
                best_f1 = f1
                improved = True

            if not improved:
                no_improve_epochs += 1
                if no_improve_epochs >= hp.early_stopping_patience:
                    print("🛑 Early stopping.")
                    break
            else:
                no_improve_epochs = 0


        # ───────────────────────────────────────────────
        #  Post-training evaluation
        # ───────────────────────────────────────────────
        plot_train_val_losses(
            train_losses, val_losses,
            save_path=os.path.join(save_dir, "loss_curve.png")
        )
        pd.DataFrame(metrics_history).to_csv(
            os.path.join(save_dir, "epoch_metrics.csv"), index=False
        )

        best_thr = 0.5
        if hp.run_threshold_optimization:
            best_thr, *_ = compute_best_threshold(model, val_ds)

        # Visualize one inference example (uses last model, just for visualization)
        feat_batch, lab_batch = next(iter(test_ds))
        show_inference(
            12, feat_batch, lab_batch, model, best_thr,
            os.path.join(save_dir, "inference.png")
        )

        # ───────────────────────────────────────────────
        #  Evaluate all three best models
        # ───────────────────────────────────────────────
        test_results = {}

        for metric_name, ckpt_name in [
            ("best_pr_auc", "best_model_pr_auc.pth"),
            ("best_iou", "best_model_iou.pth"),
            ("best_f1", "best_model_f1.pth"),
        ]:
            ckpt_path = os.path.join(save_dir, ckpt_name)
            if os.path.exists(ckpt_path):
                print(f"🔁 Loading checkpoint: {ckpt_name}")
                model.load_state_dict(torch.load(ckpt_path))
                model.eval()

                iou_t, auc_t, prec_t, rec_t, pr_auc_t, f1_t = evaluate_model(
                    model, test_ds, hp, best_thr
                )

                test_results[metric_name] = dict(
                    checkpoint=ckpt_name,
                    test_pr_auc=pr_auc_t,
                    test_auc=auc_t,
                    test_prec=prec_t,
                    test_rec=rec_t,
                    test_iou=iou_t,
                    test_f1=f1_t,
                )

                print(
                    f"📊  Test ({metric_name}) → "
                    f"PR-AUC={pr_auc_t:.4f} | AUC={auc_t:.4f} | "
                    f"Prec={prec_t:.4f} | Rec={rec_t:.4f} | "
                    f"IoU={iou_t:.4f} | F1={f1_t:.4f}"
                )
            else:
                print(f"⚠️  Skipping missing checkpoint: {ckpt_name}")

        # ───────────────────────────────────────────────
        #  Save full summary
        # ───────────────────────────────────────────────
        test_report = dict(
            sigma_profile=profile,
            combine_mode=combine_mode,
            best_threshold=best_thr,
            best_metrics=dict(
                pr_auc=best_metrics_pr,
                iou=best_metrics_iou,
                f1=best_metrics_f1,
            ),
            test_results=test_results,
            model_summary_path="model_summary.txt",
            config_path="config.json",
        )

        summary_path = os.path.join(save_dir, "summary.json")
        with open(summary_path, "w") as f:
            json.dump(test_report, f, indent=2)

        # ───────────────────────────────────────────────
        #  Final console summary
        # ───────────────────────────────────────────────
        print(f"\n📊  FINAL SUMMARY — σ-profile: {profile} | combine: {combine_mode}")
        print(
            f"   Best (validation) → "
            f"PR-AUC={best_pr_auc:.4f} | IoU={best_iou:.4f} | F1={best_f1:.4f}"
        )
        for metric_name, result in test_results.items():
            print(
                f"   [{metric_name}] Test → "
                f"PR-AUC={result['test_pr_auc']:.4f} | "
                f"IoU={result['test_iou']:.4f} | F1={result['test_f1']:.4f}"
            )
        print(
            "✅  Saved model summary, config, and metrics for "
            f"{profile}-{combine_mode}\n" + "―" * 60
        )

        # ───────────────────────────────────────────────
        #  Append to sweep summary CSV
        # ───────────────────────────────────────────────
        if test_results:
            with open(summary_csv, "a") as f:
                for metric_name, r in test_results.items():
                    f.write(
                        f"{profile},{combine_mode},{metric_name},{r.get('checkpoint','')},"
                        f"{r['test_auc']:.4f},{r['test_pr_auc']:.4f},"
                        f"{r['test_prec']:.4f},{r['test_rec']:.4f},"
                        f"{r['test_iou']:.4f},{r['test_f1']:.4f}\n"
                    )

