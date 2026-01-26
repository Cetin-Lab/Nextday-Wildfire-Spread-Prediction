# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Custom metrics for TensorFlow."""

from typing import Sequence, Optional
import tensorflow as tf


class AUCWithMaskedClass(tf.keras.metrics.AUC):
  """Computes AUC while ignoring class with id equal to `-1`.

  Assumes binary `{0, 1}` classes with a masked `{-1}` class.
  """

  def __init__(self, with_logits = False, **kwargs):
    super(AUCWithMaskedClass, self).__init__(**kwargs)
    self.with_logits = with_logits

  @tf.autograph.experimental.do_not_convert
  def update_state(self,
                   y_true,
                   y_pred,
                   sample_weight = None):
    """Accumulates metric statistics.

    `y_true` and `y_pred` should have the same shape.

    Args:
      y_true: Ground truth values.
      y_pred: Predicted values.
      sample_weight: Input value is ignored. Parameter present to match
        signature with parent class where mask `{-1}` is the sample weight.
    Returns: `None`
    """
    if self.with_logits:
      y_pred = tf.math.sigmoid(y_pred)
    mask = tf.cast(tf.not_equal(y_true, -1), tf.float32)
    super(AUCWithMaskedClass, self).update_state(
        y_true, y_pred, sample_weight=mask)


class PrecisionWithMaskedClass(tf.keras.metrics.Precision):
  """Computes precision while ignoring class with id equal to `-1`.

  Assumes binary `{0, 1}` classes with a masked `{-1}` class.
  """

  def __init__(self, with_logits = False, **kwargs):
    super(PrecisionWithMaskedClass, self).__init__(**kwargs)
    self.with_logits = with_logits

  @tf.autograph.experimental.do_not_convert
  def update_state(self,
                   y_true,
                   y_pred,
                   sample_weight = None):
    """Accumulates metric statistics.

    `y_true` and `y_pred` should have the same shape.

    Args:
      y_true: Ground truth values.
      y_pred: Predicted values.
      sample_weight: Input value is ignored. Parameter present to match
        signature with parent class where mask `{-1}` is the sample weight.
    Returns: `None`
    """
    if self.with_logits:
      y_pred = tf.math.sigmoid(y_pred)
    mask = tf.cast(tf.not_equal(y_true, -1), tf.float32)
    super(PrecisionWithMaskedClass, self).update_state(
        y_true, y_pred, sample_weight=mask)


class RecallWithMaskedClass(tf.keras.metrics.Recall):
  """Computes recall while ignoring class with id equal to `-1`.

  Assumes binary `{0, 1}` classes with a masked `{-1}` class.
  """

  def __init__(self, with_logits = False, **kwargs):
    super(RecallWithMaskedClass, self).__init__(**kwargs)
    self.with_logits = with_logits

  @tf.autograph.experimental.do_not_convert
  def update_state(self,
                   y_true,
                   y_pred,
                   sample_weight = None):
    """Accumulates metric statistics.

    `y_true` and `y_pred` should have the same shape.

    Args:
      y_true: Ground truth values.
      y_pred: Predicted values.
      sample_weight: Input value is ignored. Parameter present to match
        signature with parent class where mask `{-1}` is the sample weight.
    Returns: `None`
    """
    if self.with_logits:
      y_pred = tf.math.sigmoid(y_pred)
    mask = tf.cast(tf.not_equal(y_true, -1), tf.float32)
    super(RecallWithMaskedClass, self).update_state(
        y_true, y_pred, sample_weight=mask)
    ##################################### added functions 
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import precision_score, recall_score, f1_score

def compute_best_threshold(model, dataset, thresholds=np.linspace(0.05, 0.95, 19), save_plot_path=None):
    best_f1 = -1
    best_threshold = 0.5
    best_prec = 0.0
    best_rec = 0.0

    all_y_true, all_y_pred = [], []

    model.eval()
    with torch.no_grad():
        for feats_tf, labs_tf in dataset:
            x = torch.from_numpy(feats_tf.numpy()).permute(0, 3, 1, 2).float().to(next(model.parameters()).device)
            preds = model(x).squeeze(1).detach().cpu().numpy()
            targets = labs_tf.numpy()[..., 0]

            mask = targets != -1
            all_y_true.append(targets[mask])
            all_y_pred.append(preds[mask])

    y_true = np.concatenate(all_y_true)
    y_pred = np.concatenate(all_y_pred)

    f1_scores = []
    for t in thresholds:
        bin_pred = (y_pred > t).astype(np.uint8)
        prec = precision_score(y_true, bin_pred, zero_division=0)
        rec  = recall_score(y_true, bin_pred, zero_division=0)
        f1   = f1_score(y_true, bin_pred, zero_division=0)
        f1_scores.append(f1)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t
            best_prec = prec
            best_rec = rec

    if save_plot_path:
        plt.figure(figsize=(6, 4))
        plt.plot(thresholds, f1_scores, marker='o')
        plt.title("F1-Score vs Threshold")
        plt.xlabel("Threshold")
        plt.ylabel("F1-Score")
        plt.grid(True)
        plt.savefig(save_plot_path)
        print(f"📈 Saved F1-score vs threshold plot to {save_plot_path}")
        plt.close()

    return best_threshold, best_f1, best_prec, best_rec


def masked_iou(y_true, y_pred, ignore_val=-1):
    mask = y_true != ignore_val
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    inter = np.logical_and(y_true, y_pred).sum()
    union = np.logical_or(y_true, y_pred).sum()
    return inter / union if union else 0.0

from sklearn.metrics import precision_recall_curve, auc

def compute_pr_auc(y_true, y_pred, ignore_val=-1):
    mask = y_true != ignore_val
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    return auc(recall, precision)
