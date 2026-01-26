# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 02:09:31 2025

@author: olivi
"""

# ───────────────────── cnn_autoencoder_gaussian_sweep.py ─────────────────────
"""
Train the TF CNN-autoencoder once per Gaussian σ-profile (PrevFireMask softening).
Keeps all helper functions (loss curves, inference gallery, etc.).
Results stored in:  <hp.model_dir>/<profile_name>/
"""
# train_tf_cnn_gaussian.py
# ───────────────────────────────────────────────────────────────
import os, sys, re, json, numpy as np, tensorflow as tf
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ── Gaussian σ-profiles -----------------------------------------
import prefire_gaussian                 # defines SIGMA_PROFILES

# ── Project imports ---------------------------------------------
sys.path += [
    r"C:\Users\Dell\Desktop\Shuaiang\wildfire_detection\codetfAE1",
    r"C:\Users\Dell\Desktop\Shuaiang\wildfire_detection\codetfAE1/models",
]
from dataset_adjust_pre_post_gaussian_B import make_dataset      # reads hp.gaussian_profile
from cnn_autoencoder_model import create_model
from constants import INPUT_FEATURES
from losses import weighted_cross_entropy_with_logits_with_masked_class
from metrics import (AUCWithMaskedClass,
                     PrecisionWithMaskedClass,
                     RecallWithMaskedClass,
                     masked_iou)

# ── Global hyper-params template ---------------------------------
class HParams:
    train_path = "C:/Users/Dell/Desktop/Shuaiang/wildfire_detection/archive/next_day_wildfire_spread_train_*.tfrecord"
    eval_path  = "C:/Users/Dell/Desktop/Shuaiang/wildfire_detection/archive/next_day_wildfire_spread_eval_*.tfrecord"
    test_path =  "C:/Users/Dell/Desktop/Shuaiang/wildfire_detection/archive/next_day_wildfire_spread_test_*.tfrecord"
    input_features       = list(INPUT_FEATURES)
    output_features      = ["FireMask"]
    data_sample_size     = 64
    sample_size          = 64
    output_sample_size   = 64
    batch_size           = 32
    shuffle_buffer_size  = 500
    compression_type     = ""

    random_flip   = True
    random_rotate = False
    random_crop   = False

    input_sequence_length  = 1
    output_sequence_length = 1
    binarize_output        = True
    downsample_threshold   = 0.3

    encoder_layers = [16, 32, 32]
    decoder_layers = [32, 32, 16]
    encoder_pools  = [1, 2, 2]
    decoder_pools  = [2, 2, 2]
    dropout        = 0.1
    batch_norm     = "all"
    l1_reg         = 0.0
    l2_reg         = 1e-5

    # required by dataset.make_dataset
    azimuth_in_channel  = "th"
    azimuth_out_channel = None

    learning_rate   = 1e-4
    epochs          = 100
    steps_per_epoch = 1000
    pos_weight      = 3.0

    # Gaussian profile (overwritten inside sweep)
    gaussian_profile = "moderate"
    gaussian_combine = "union"

    root_out = r"C:/Users/Dell/Desktop/Shuaiang/wildfire_detection/output_model"

# ───────────────── helpers ───────────────────────────────────
def ensure_dir(p): os.makedirs(p, exist_ok=True); return p

def plot_losses(tr, vl, fname):
    plt.figure(figsize=(6,3))
    plt.plot(tr,label="Train"); plt.plot(vl,label="Val")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.grid(); plt.legend()
    plt.tight_layout(); plt.savefig(fname); plt.close(); print("📉",fname)

def gallery(model, ds, fname, rows=20, thr=.5):
    feats, gts = next(iter(ds))
    preds = (tf.sigmoid(model(feats,False))>thr).numpy()[...,0]
    cmap = mcolors.ListedColormap(["black","silver","orangered"])
    norm = mcolors.BoundaryNorm([-1,-.1,.001,1], cmap.N)
    plt.figure(figsize=(15,4*rows))
    for i in range(rows):
        for j,img in enumerate([feats[i,...,-1], gts[i,...,0], preds[i]]):
            plt.subplot(rows,3,3*i+j+1); plt.imshow(img,cmap=cmap,norm=norm)
            plt.axis("off"); plt.title(["Prev","GT","Pred"][j])
    plt.tight_layout(); plt.savefig(fname); plt.close(); print("🖼️",fname)

def masked_iou_tf(model, ds, steps, thr=.5):
    vals=[]
    for feats,labs in ds.take(steps):
        p=(tf.sigmoid(model(feats,False))>thr).numpy()[...,0]
        g=labs[...,0].numpy()
        vals += [masked_iou(gb,pb) for gb,pb in zip(g,p)]
    return float(np.mean(vals))


# ── Progress bar: ONE line per epoch ─────────────────────────────
# ── Progress bar: “epoch k/⌿, batch i/1000” on ONE line ───────────
class EpochBatchBar(tf.keras.callbacks.Callback):
    """
    Prints one tqdm bar per epoch:
       epoch 1/100:  17%|███▋        | 170/1000 [loss=0.3124 val=0.2987 auc=0.781]
    Spyder/Jupyter-friendly (no extra lines per batch).
    """
    def __init__(self, steps_per_epoch: int, total_epochs: int):
        super().__init__()
        self.steps = steps_per_epoch
        self.total_epochs = total_epochs
        self.bar = None

    # new bar each epoch, sized to batches
    def on_epoch_begin(self, epoch, logs=None):
        desc = f"epoch {epoch+1}/{self.total_epochs}"
        self.bar = tqdm(total=self.steps, desc=desc,
                        unit="batch", dynamic_ncols=True, leave=True)

    # update once per batch
    def on_train_batch_end(self, batch, logs=None):
        self.bar.update(1)

    # close bar, show metrics
    def on_epoch_end(self, epoch, logs=None):
        self.bar.set_postfix({
            "loss"    : f"{logs['loss']:.4f}",
            "val_loss": f"{logs['val_loss']:.4f}",
            "val_auc" : f"{logs['val_auc']:.3f}"
        })
        self.bar.close()



# ── Saver callback (best / last) ─────────────────────────────
class Saver(tf.keras.callbacks.Callback):
    def __init__(self, out_dir):
        super().__init__(); self.d=out_dir; self.best=-1
    def _dump(self, tag, logs, epoch):
        self.model.save_weights(os.path.join(self.d,f"{tag}.h5"))
        with open(os.path.join(self.d,f"{tag}_metrics.json"),"w") as f:
            json.dump({**logs,"epoch":epoch+1}, f, indent=2)
    def on_epoch_end(self, epoch, logs=None):
        self._dump("last", logs, epoch)
        if logs["val_auc"] > self.best:
            self.best = logs["val_auc"]
            self._dump("best", logs, epoch)
            print(f"   🔏 best.h5 updated (AUC={self.best:.3f})")

# ───────────────── sweep over σ-profiles ─────────────────────
for profile in prefire_gaussian.SIGMA_PROFILES:
    print(f"\n🚀  profile **{profile}**")
    hp = HParams(); hp.gaussian_profile = profile
    run_dir = ensure_dir(os.path.join(hp.root_out, profile))
    print("📂 outputs →", run_dir)

    # datasets
    train_ds = make_dataset(hp, mode="train")
    val_ds   = make_dataset(hp, mode="eval")
    test_ds  = make_dataset(hp, mode="predict")

    # model
    inp = tf.keras.Input((hp.sample_size,hp.sample_size,len(hp.input_features)))
    out = create_model(inp, 1, hp.encoder_layers, hp.decoder_layers,
                       hp.encoder_pools, hp.decoder_pools,
                       dropout=hp.dropout, batch_norm=hp.batch_norm,
                       l2_regularization=hp.l2_reg)
    model = tf.keras.Model(inp,out)
    model.compile(
        tf.keras.optimizers.Adam(hp.learning_rate),
        weighted_cross_entropy_with_logits_with_masked_class(pos_weight=hp.pos_weight),
        metrics=[AUCWithMaskedClass(with_logits=True,name="val_auc"),
                 PrecisionWithMaskedClass(with_logits=True,name="precision"),
                 RecallWithMaskedClass(with_logits=True,name="recall")]
    )

    # training
    hist = model.fit(
        train_ds, validation_data=val_ds,
        epochs=hp.epochs,
        steps_per_epoch=hp.steps_per_epoch,
        validation_steps=hp.steps_per_epoch,
        verbose=0,
        callbacks=[
            EpochBatchBar(steps_per_epoch=hp.steps_per_epoch,
                          total_epochs=hp.epochs),
            Saver(run_dir)
        ]
    )

    # test metrics
    model.load_weights(os.path.join(run_dir,"best.h5"))
    test_log = model.evaluate(test_ds, steps=hp.steps_per_epoch,
                              return_dict=True, verbose=0)
    test_log["masked_iou"] = masked_iou_tf(model,test_ds,hp.steps_per_epoch,.5)
    p,r = test_log["precision"], test_log["recall"]
    test_log["f1"] = 2*p*r/(p+r+1e-8)
    with open(os.path.join(run_dir,"test_metrics.json"),"w") as f:
        json.dump(test_log, f, indent=2)
    for k,v in test_log.items(): print(f"{k:12}: {v:.4f}")

    # artefacts
    plot_losses(hist.history["loss"], hist.history["val_loss"],
                os.path.join(run_dir,"loss_curve.png"))
    gallery(model, test_ds,
            os.path.join(run_dir,"inference.png"), rows=20, thr=.5)