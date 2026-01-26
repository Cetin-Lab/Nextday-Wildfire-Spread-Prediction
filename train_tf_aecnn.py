# -*- coding: utf-8 -*-
"""
Created on Mon Apr 21 17:01:59 2025

@author: olivi
"""
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CNN‑autoencoder training with masked weighted BCE, tqdm bar,
best.pt/best_metrics.json and last.pt/last_metrics.json.
"""

# ───────────────────────────── imports ─────────────────────────────
import os, sys, json, tensorflow as tf
from tqdm.auto import tqdm                      # single-line bars in Spyder/Jupyter
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ─── project paths ─────────────────────────────────────────────────
sys.path += [
    r"D:/wildfire/wildfire_detection/codetfAE1",
    r"D:/wildfire/wildfire_detection/codetfAE1/models",
]

#from dataset_adjust_pre_post import make_dataset
from dataset import make_dataset
from constants import INPUT_FEATURES
from hadamard_cnn_autoencoder_model import create_model
from losses import weighted_cross_entropy_with_logits_with_masked_class
from metrics import (AUCWithMaskedClass,
                     PrecisionWithMaskedClass,
                     RecallWithMaskedClass,
                     masked_iou)

# ─────────────────────── hyper-parameters ──────────────────────────
# ───── hyper‑params ─────────────────────────────────────────────────
class HParams:
    train_path = "D:/wildfire/wildfire_detection/archive/next_day_wildfire_spread_train_*.tfrecord"
    eval_path  = "D:/wildfire/wildfire_detection/archive/next_day_wildfire_spread_eval_*.tfrecord"
    test_path =  "D:/wildfire/wildfire_detection/archive/next_day_wildfire_spread_test_*.tfrecord"
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

    model_dir = "D:/wildfire/wildfire_detection/codetfAE1/AECNN/hada_AECNN"
hp = HParams()
os.makedirs(hp.model_dir, exist_ok=True)


# ───────────────────────── datasets ────────────────────────────────
train_ds = make_dataset(hp, mode="train")
val_ds   = make_dataset(hp, mode="eval")
test_ds  = make_dataset(hp, mode="predict")

# ────────────────────────── model  ─────────────────────────────────
inp = tf.keras.Input(
    shape=(hp.sample_size, hp.sample_size, len(hp.input_features))
)
out = create_model(
    inp,
    num_out_channels=1,
    encoder_layers=hp.encoder_layers,
    decoder_layers=hp.decoder_layers,
    encoder_pools=hp.encoder_pools,
    decoder_pools=hp.decoder_pools,
    dropout=hp.dropout,
    batch_norm=hp.batch_norm,
    l2_regularization=hp.l2_reg,
)
model = tf.keras.Model(inp, out)
model.compile(
    optimizer=tf.keras.optimizers.Adam(hp.learning_rate),
    loss=weighted_cross_entropy_with_logits_with_masked_class(pos_weight=hp.pos_weight),
    metrics=[
        AUCWithMaskedClass(with_logits=True, name="val_auc"),
        PrecisionWithMaskedClass(with_logits=True, name="precision"),
        RecallWithMaskedClass(with_logits=True, name="recall"),
        
    ],
)

# ─────────────── callbacks ────────────────────────────────────────
class TqdmBar(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.bar = tqdm(total=hp.steps_per_epoch,
                        desc=f"Epoch {epoch+1}/{hp.epochs}",
                        unit="batch",
                        leave=False)
    def on_batch_end(self, batch, logs=None):
        self.bar.update(1)
        self.bar.set_postfix(loss=f"{logs['loss']:.4f}")
    def on_epoch_end(self, epoch, logs=None):
        self.bar.close()
        print(f"\nEpoch {epoch+1}: "
              f"loss={logs['loss']:.4f}  "
              f"val_loss={logs['val_loss']:.4f}  "
              f"val_auc={logs['val_auc']:.3f}")

class SaveCheckpoints(tf.keras.callbacks.Callback):
    def __init__(self, out_dir):
        super().__init__()
        self.dir = out_dir
        self.best_auc = -float("inf")

    def on_epoch_end(self, epoch, logs=None):
        # last
        self.model.save_weights(os.path.join(self.dir, "last.weights.h5"))

        with open(os.path.join(self.dir, "last_metrics.json"), "w") as f:
            json.dump(dict(epoch=epoch+1, **logs), f, indent=2)
        # best
        if logs["val_auc"] > self.best_auc:
            self.best_auc = logs["val_auc"]
            self.model.save_weights(os.path.join(self.dir, "best.weights.h5"))

            with open(os.path.join(self.dir, "best_metrics.json"), "w") as f:
                json.dump(dict(epoch=epoch+1, **logs), f, indent=2)
            print(f"🔏 best.pt updated  (AUC={self.best_auc:.3f})")

# ───────────────────────── training ───────────────────────────────
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=hp.epochs,
    steps_per_epoch=hp.steps_per_epoch,
    validation_steps=hp.steps_per_epoch,     # required: val_ds is infinite
    callbacks=[TqdmBar(), SaveCheckpoints(hp.model_dir)],
    verbose=1,
)

# ─────────────────────── test evaluation ──────────────────────────
print("\n🧪 Test-set evaluation (best.pt)")
model.load_weights(os.path.join(hp.model_dir, "best.weights.h5"))

test_logs = model.evaluate(test_ds,
                           steps=hp.steps_per_epoch,   # dataset repeats
                           return_dict=True,
                           verbose=1)
for k, v in test_logs.items():
    print(f"{k:12s}: {v:.4f}")

import numpy as np

def compute_masked_iou(model, dataset, steps, thr=0.5):
    ious = []
    for i, (feats, labels) in enumerate(dataset.take(steps)):
        logits = model(feats, training=False)
        preds  = tf.sigmoid(logits) > thr         # (B,H,W,1) bool
        y_true = labels[..., 0].numpy()
        y_pred = preds[..., 0].numpy()
        for b in range(y_true.shape[0]):
            ious.append(masked_iou(y_true[b], y_pred[b]))
    return float(np.mean(ious))

test_iou = compute_masked_iou(model, test_ds,
                              steps=hp.steps_per_epoch,
                              thr=0.5)

# precision & recall already in `test_logs`
precision = test_logs["precision"]
recall    = test_logs["recall"]
test_f1   = 2 * precision * recall / (precision + recall + 1e-8)

print(f"masked IoU  : {test_iou:.4f}")
print(f"F1-score    : {test_f1:.4f}")

# append and save to JSON
test_logs.update({"masked_iou": test_iou, "f1": test_f1})
with open(os.path.join(hp.model_dir, "test_metrics.json"), "w") as f:
    json.dump(test_logs, f, indent=2)
print(f"📝 test metrics saved to {hp.model_dir}/test_metrics.json")
# ─────────────────────── loss curves  ─────────────────────────────
plt.figure(figsize=(8,5))
plt.plot(history.history["loss"],     label="Train")
plt.plot(history.history["val_loss"], label="Val")
plt.title("Loss curves");  plt.xlabel("Epoch");  plt.ylabel("Loss")
plt.grid(True);  plt.legend()
png_loss = os.path.join(hp.model_dir, "loss_curve.png")
plt.tight_layout(); plt.savefig(png_loss); plt.show()
print(f"📉 loss curve saved to {png_loss}")

# ─────────────────── inference gallery  ───────────────────────────
def visualize_predictions(dataset, rows=6, thr=0.5):
    cmap = mcolors.ListedColormap(["black", "silver", "orangered"])
    norm = mcolors.BoundaryNorm([-1, -0.1, 0.001, 1], cmap.N)
    feats, labels = next(iter(dataset))
    preds = (tf.sigmoid(model(feats, training=False)) > thr).numpy()[...,0]

    plt.figure(figsize=(15, 4*rows))
    for i in range(rows):
        # prev-day
        plt.subplot(rows,3,3*i+1); plt.imshow(feats[i,...,-1], cmap=cmap, norm=norm)
        plt.title("Prev-day"); plt.axis("off")
        # ground truth
        plt.subplot(rows,3,3*i+2); plt.imshow(labels[i,...,0], cmap=cmap, norm=norm)
        plt.title("Ground Truth"); plt.axis("off")
        # prediction
        plt.subplot(rows,3,3*i+3); plt.imshow(preds[i], cmap=cmap, norm=norm)
        plt.title("Prediction"); plt.axis("off")
    plt.tight_layout()
    out = os.path.join(hp.model_dir, "inference_example.png")
    plt.savefig(out); plt.show()
    print(f"🖼️ inference example saved to {out}")

visualize_predictions(test_ds, rows=20, thr=0.5)

# import os, sys, json, tensorflow as tf
# from tqdm import tqdm

# # ───── simple ModeKeys enum ─────────────────────────────────────────
# class ModeKeys:
#     TRAIN, EVAL, PREDICT = "train", "eval", "predict"

# # ───── project imports ──────────────────────────────────────────────
# sys.path += ["C:/Users/Dell/Desktop/Shuaiang/wildfire_detection/code",
#              "C:/Users/Dell/Desktop/Shuaiang/wildfire_detection/code/models"]
# from dataset_adjust_post import make_dataset
# from constants import INPUT_FEATURES
# from cnn_autoencoder_model import create_model
# from losses import weighted_cross_entropy_with_logits_with_masked_class
# from metrics import (
#     AUCWithMaskedClass, PrecisionWithMaskedClass,
#     RecallWithMaskedClass
# )

# # ───── hyper‑params ─────────────────────────────────────────────────
# class HParams:
#     train_path = "C:/Users/Dell/Desktop/Shuaiang/wildfire_detection/archive/next_day_wildfire_spread_train_*.tfrecord"
#     eval_path  = "C:/Users/Dell/Desktop/Shuaiang/wildfire_detection/archive/next_day_wildfire_spread_eval_*.tfrecord"
#     test_path =  "C:/Users/Dell/Desktop/Shuaiang/wildfire_detection/archive/next_day_wildfire_spread_test_*.tfrecord"
#     input_features       = list(INPUT_FEATURES)
#     output_features      = ["FireMask"]
#     data_sample_size     = 64
#     sample_size          = 64
#     output_sample_size   = 64
#     batch_size           = 32
#     shuffle_buffer_size  = 500
#     compression_type     = ""

#     random_flip   = True
#     random_rotate = False
#     random_crop   = False

#     input_sequence_length  = 1
#     output_sequence_length = 1
#     binarize_output        = True
#     downsample_threshold   = 0.3

#     encoder_layers = [16, 32, 32]
#     decoder_layers = [32, 32, 16]
#     encoder_pools  = [1, 2, 2]
#     decoder_pools  = [2, 2, 2]
#     dropout        = 0.1
#     batch_norm     = "all"
#     l1_reg         = 0.0
#     l2_reg         = 1e-5

#     # required by dataset.make_dataset
#     azimuth_in_channel  = "th"
#     azimuth_out_channel = None

#     learning_rate   = 1e-4
#     epochs          = 100
#     steps_per_epoch = 1000
#     pos_weight      = 3.0

#     model_dir = "C:/Users/Dell/Desktop/Shuaiang/wildfire_detection//output_model"
# hp = HParams()
# os.makedirs(hp.model_dir, exist_ok=True)

# # ───── datasets ─────────────────────────────────────────────────────
# train_ds = make_dataset(hp, mode=ModeKeys.TRAIN)
# val_ds   = make_dataset(hp, mode=ModeKeys.EVAL)

# # ───── model ────────────────────────────────────────────────────────
# inp = tf.keras.Input(
#     shape=(hp.sample_size, hp.sample_size, len(hp.input_features))
# )
# out = create_model(
#     inp,
#     num_out_channels=1,
#     encoder_layers=hp.encoder_layers,
#     decoder_layers=hp.decoder_layers,
#     encoder_pools=hp.encoder_pools,
#     decoder_pools=hp.decoder_pools,
#     dropout=hp.dropout,
#     batch_norm=hp.batch_norm,
#     l1_regularization=hp.l1_reg,
#     l2_regularization=hp.l2_reg,
# )

# model = tf.keras.Model(inp, out)
# model.compile(
#     optimizer=tf.keras.optimizers.Adam(hp.learning_rate),
#     loss=weighted_cross_entropy_with_logits_with_masked_class(
#         pos_weight=hp.pos_weight
#     ),
#     metrics=[
#         AUCWithMaskedClass(with_logits=True, name="val_auc"),
#         PrecisionWithMaskedClass(with_logits=True, name="precision"),
#         RecallWithMaskedClass(with_logits=True, name="recall"),
#     ],
# )

# # ───── tqdm progress‑bar callback ───────────────────────────────────
# class TqdmBar(tf.keras.callbacks.Callback):
#     def on_epoch_begin(self, epoch, logs=None):
#         self.bar = tqdm(
#             total=hp.steps_per_epoch,
#             desc=f"Epoch {epoch+1}/{hp.epochs}",
#             unit="batch"
#         )
#     def on_batch_end(self, batch, logs=None):
#         self.bar.update(1)
#         self.bar.set_postfix(loss=f"{logs['loss']:.4f}")
#     def on_epoch_end(self, epoch, logs=None):
#         self.bar.close()
#         print(
#             f"\nEpoch {epoch+1}: "
#             f"loss={logs['loss']:.4f}, "
#             f"val_loss={logs['val_loss']:.4f}, "
#             f"val_auc={logs['val_auc']:.3f}"
#         )

# # ───── checkpoint + metric saver ───────────────────────────────────
# class SaveCheckpoints(tf.keras.callbacks.Callback):
#     def __init__(self, out_dir):
#         super().__init__()
#         self.dir = out_dir
#         self.best_auc = -float("inf")
#         self.best_logs = None

#     def on_epoch_end(self, epoch, logs=None):
#         # always save "last" checkpoint
#         last = dict(epoch=epoch+1, **logs)
#         self.model.save_weights(os.path.join(self.dir, "last.pt"))
#         with open(os.path.join(self.dir, "last_metrics.json"), "w") as f:
#             json.dump(last, f, indent=2)

#         # save "best" if val_auc improves
#         if logs["val_auc"] > self.best_auc:
#             self.best_auc = logs["val_auc"]
#             self.best_logs = last
#             self.model.save_weights(os.path.join(self.dir, "best.pt"))
#             with open(os.path.join(self.dir, "best_metrics.json"), "w") as f:
#                 json.dump(self.best_logs, f, indent=2)

#     def on_train_end(self, logs=None):
#         print("\n=== Final metrics (last.pt) ===")
#         print(json.dumps(
#             json.load(open(os.path.join(self.dir, "last_metrics.json"))),
#             indent=2
#         ))
#         print("\n=== Best metrics  (best.pt) ===")
#         print(json.dumps(
#             json.load(open(os.path.join(self.dir, "best_metrics.json"))),
#             indent=2
#         ))

# # ───── train ───────────────────────────────────────────────────────
# model.fit(
#     train_ds,
#     validation_data=val_ds,
#     epochs=hp.epochs,
#     steps_per_epoch=hp.steps_per_epoch,
#     validation_steps=hp.steps_per_epoch, 
#     callbacks=[TqdmBar(), SaveCheckpoints(hp.model_dir)],
#     verbose=0,
# )
# # ──