# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 01:50:32 2025

@author: olivi
"""

# ──────────────────────────────────────────────────────────────
#  PyTorch training pipeline – sweep over Gaussian σ profiles
# ──────────────────────────────────────────────────────────────
import os, sys, re, json

# ------------------------------------------------------------------
# 1.  Paths / imports
# ------------------------------------------------------------------
sys.path += ["D:/wildfire/wildfire_detection/codetfAE1",
             "D:/wildfire/wildfire_detection/codetfAE1/models"]
os.environ.update({
    "OMP_NUM_THREADS": "1",
    "TF_NUM_INTRAOP_THREADS": "1",
    "TF_NUM_INTEROP_THREADS": "1",
})

import torch, numpy as np, pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt, colors

from hadamard_unet_BN_dropout_DEEP_v0 import HadamardUnet
from dataset_adjust_pre_post_gaussian_B import make_dataset, ModeKeys
from constants import INPUT_FEATURES
import prefire_gaussian                                  # ← profiles live here
from losses1 import combo_loss
from metrics import (AUCWithMaskedClass, PrecisionWithMaskedClass,
                     RecallWithMaskedClass, masked_iou,
                     compute_best_threshold)

# ------------------------------------------------------------------
# 2.  Hyper-parameter container
# ------------------------------------------------------------------
class HParams:
    # TFRecord paths
    train_path = "D:/wildfire/wildfire_detection/archive/next_day_wildfire_spread_train_*.tfrecord"
    eval_path  = "D:/wildfire/wildfire_detection/archive/next_day_wildfire_spread_eval_*.tfrecord"
    test_path  = "D:/wildfire/wildfire_detection/archive/next_day_wildfire_spread_test_*.tfrecord"

    # I/O features
    input_features  = list(INPUT_FEATURES)
    output_features = ["FireMask"]
    data_sample_size = 64
    sample_size      = 64
    output_sample_size = 64
    input_sequence_length = 1
    output_sequence_length = 1
    azimuth_in_channel  = "th"
    azimuth_out_channel = None

    # Data pipeline options
    shuffle_buffer_size = 500
    compression_type   = ""
    random_flip  = True
    random_rotate = False
    random_crop   = False
    downsample_threshold = 0.3
    binarize_output = True

    # Training
    batch_size     = 32
    epochs         = 100
    steps_per_epoch= 1000
    learning_rate  = 1e-4
    pos_weight     = 3.0
    run_threshold_optimization = False

    # Gaussian softening (defaults; will be overwritten in the sweep)
    gaussian_profile = "moderate"
    gaussian_combine = "union"

# ------------------------------------------------------------------
# 3.  Utility helpers
# ------------------------------------------------------------------
def get_numbered_save_dir(root:str, prefix:str) -> str:
    """Create .../root/prefixXX where XX is an auto-incremented id."""
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
    if save_path: plt.savefig(save_path); print("📉  Saved", save_path)
    plt.close()

def show_inference(n_rows, feats_tf, labs_tf, model, thr, save_path):
    seg_cmap = colors.ListedColormap(["black", "silver", "orangered"])
    seg_norm = colors.BoundaryNorm([-1, -.1, .001, 1], seg_cmap.N)
    x = torch.from_numpy(feats_tf.numpy()).permute(0,3,1,2).float().cuda()
    with torch.no_grad():
        preds = (torch.sigmoid(model(x)).squeeze(1).cpu().numpy() > thr)
    gts  = labs_tf.numpy()[...,0]
    prev = (feats_tf.numpy()[:,:,:,-1] > .5).astype(np.int32)
    plt.figure(figsize=(12,4*n_rows))
    for i in range(n_rows):
        for j, img in enumerate([prev[i], gts[i], preds[i]]):
            plt.subplot(n_rows,3,3*i+j+1); plt.imshow(img, cmap=seg_cmap, norm=seg_norm)
            plt.axis("off"); plt.title(["Prev","GT","Pred"][j])
    plt.tight_layout()
    plt.savefig(save_path); print("🖼️  Saved", save_path); plt.close()

def evaluate_model(model, dataset, hp, threshold=0.5):
    auc   = AUCWithMaskedClass()
    prec  = PrecisionWithMaskedClass()
    rec   = RecallWithMaskedClass()
    ious  = []

    model.eval()
    with torch.no_grad():
        for i, (feats_tf, labs_tf) in enumerate(dataset.take(hp.steps_per_epoch)):
            x = torch.from_numpy(feats_tf.numpy()).permute(0, 3, 1, 2).float().cuda()
            y_pred = torch.sigmoid(model(x)).squeeze(1).cpu().numpy()
            y_true = labs_tf.numpy()[..., 0]

            auc.update_state(y_true, y_pred)
            bin_pred = (y_pred > threshold).astype(np.float32)
            prec.update_state(y_true, bin_pred)
            rec.update_state(y_true,  bin_pred)

            for j in range(len(y_true)):
                ious.append(masked_iou(y_true[j], bin_pred[j]))

    # Convert TF tensors → NumPy → Python float
    return (
        float(np.mean(ious)),
        float(auc.result().numpy()),
        float(prec.result().numpy()),
        float(rec.result().numpy()),
    )


# ------------------------------------------------------------------
# 4.  Main sweep loop
# ------------------------------------------------------------------
EXPERIMENT_ROOT = "gaussian_sweep_results"
ALL_PROFILES     = list(prefire_gaussian.SIGMA_PROFILES.keys())

for profile in ALL_PROFILES:
    print(f"\n🚀  Running experiment for σ-profile: **{profile}**\n" + "-"*60)
    hp = HParams()               # fresh copy each run
    hp.gaussian_profile = profile
    save_dir = get_numbered_save_dir(EXPERIMENT_ROOT, f"{profile}_")
    print("📂  Saving artefacts to:", save_dir)

    # ── Dataset (regenerated for this profile) ──────────────────
    train_ds = make_dataset(hp, mode=ModeKeys.TRAIN)
    val_ds   = make_dataset(hp, mode=ModeKeys.EVAL)
    test_ds  = make_dataset(hp, mode=ModeKeys.PREDICT)

    # ── Model / optimiser ───────────────────────────────────────
    model = HadamardUnet(len(hp.input_features), hp.sample_size, 1).cuda()
    optimiser = torch.optim.Adam(model.parameters(), lr=hp.learning_rate)

    best_auc, train_losses, val_losses, metrics_history = 0.0, [], [], []

    for epoch in range(1, hp.epochs+1):
        # ---- Training loop ----
        model.train(); tloss, n = 0.0, 0
        for i,(feats_tf,labs_tf) in enumerate(tqdm(train_ds.take(hp.steps_per_epoch),
                                                   desc=f"[{profile}] Epoch {epoch} • Train",
                                                   ncols=90)):
            x = torch.from_numpy(feats_tf.numpy()).permute(0,3,1,2).float().cuda()
            y = torch.from_numpy(labs_tf.numpy()).permute(0,3,1,2).float().cuda()
            optimiser.zero_grad()
            loss = combo_loss(model(x), y, pos_weight=hp.pos_weight)
            loss.backward(); optimiser.step()
            tloss += loss.item(); n += 1
        train_losses.append(tloss/n)

        # ---- Validation loop ----
        model.eval(); vloss, m = 0.0, 0
        with torch.no_grad():
            for feats_tf,labs_tf in val_ds.take(hp.steps_per_epoch):
                x = torch.from_numpy(feats_tf.numpy()).permute(0,3,1,2).float().cuda()
                y = torch.from_numpy(labs_tf.numpy()).permute(0,3,1,2).float().cuda()
                vloss += combo_loss(model(x), y, pos_weight=hp.pos_weight).item(); m+=1
        val_losses.append(vloss/m)

        # ---- Metrics ----
        iou, auc, prec, rec = evaluate_model(model, val_ds, hp)
        metrics_history.append(dict(epoch=epoch, train_loss=train_losses[-1],
                                    val_loss=val_losses[-1], iou=iou,
                                    auc=auc, precision=prec, recall=rec))
        print(f"📝  Epoch {epoch:03d} | TL {train_losses[-1]:.4f} | VL {val_losses[-1]:.4f} "
              f"| AUC {auc:.4f} | Prec {prec:.4f} | Rec {rec:.4f} | IoU {iou:.4f}")

        # ---- Check-point best model ----
        if auc > best_auc:
            best_auc = auc; torch.save(model.state_dict(),
                                       os.path.join(save_dir, "best_model.pth"))
            best_metrics = metrics_history[-1]
            print(f"   💾 New best model saved (AUC={best_auc:.4f})")

    # ------------------------------------------------------------------
    # 5.  Post-training artefacts
    # ------------------------------------------------------------------
    plot_train_val_losses(train_losses, val_losses,
                          save_path=os.path.join(save_dir,"loss_curve.png"))
    pd.DataFrame(metrics_history).to_csv(os.path.join(save_dir,"epoch_metrics.csv"),index=False)

    # Threshold search (optional)
    if hp.run_threshold_optimization:
        best_thr, *_ = compute_best_threshold(model, val_ds)
    else:
        best_thr = 0.5

    # Quick qualitative inference on test batch
    feat_batch, lab_batch = next(iter(test_ds))
    show_inference(12, feat_batch, lab_batch, model, best_thr,
                   os.path.join(save_dir,"inference.png"))

    # Final test metrics
    iou_t, auc_t, prec_t, rec_t = evaluate_model(model, test_ds, hp, best_thr)
    f1_t  = 2*prec_t*rec_t/(prec_t+rec_t+1e-8)
    test_report = dict(best_metrics, test_auc=auc_t, test_prec=prec_t,
                       test_rec=rec_t, test_iou=iou_t, test_f1=f1_t,
                       best_threshold=best_thr)
    with open(os.path.join(save_dir,"summary.json"),"w") as f:
        json.dump(test_report, f, indent=2)
        # ── Human-readable console recap ──────────────────────────────
        print("\n📊  FINAL SUMMARY — σ-profile:", profile)
        print("   Best-epoch (val)  →  "
              f"AUC={best_metrics['auc']:.4f}  "
              f"Prec={best_metrics['precision']:.4f}  "
              f"Rec={best_metrics['recall']:.4f}  "
              f"IoU={best_metrics['iou']:.4f}  "
              f"Loss={best_metrics['val_loss']:.4f}")
        print("   Test set (thr={:.2f}) →  "
              f"AUC={auc_t:.4f}  "
              f"Prec={prec_t:.4f}  "
              f"Rec={rec_t:.4f}  "
              f"IoU={iou_t:.4f}  "
              f"F1={f1_t:.4f}".format(best_thr))
        print("   Artefacts saved in:", save_dir, "\n" + "―"*60)

    print("✅  Saved summary for", profile, "→", save_dir)
