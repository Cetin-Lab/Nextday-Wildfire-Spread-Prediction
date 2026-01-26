# -*- coding: utf-8 -*-
"""
Created on Fri May 16 17:13:31 2025

@author: Dell
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.linalg import hadamard

def weighted_bce_with_logits_masked(y_pred, y_true, pos_weight=1.0):
    """
    Implements: mask * weighted_cross_entropy_with_logits
    Ignores targets with value -1 (uncertain).
    """
    mask = (y_true != -1).float()
    loss = F.binary_cross_entropy_with_logits(
        input=y_pred,
        target=y_true,
        weight=mask * (pos_weight * y_true + (1.0 - y_true)),
        reduction='sum'
    )
    return loss / mask.sum().clamp(min=1.0)  # prevent div by 0

########################

def dice_loss(preds, targets, smooth=1e-6):
    """
    Dice loss for binary segmentation, ignoring uncertain targets (-1).
    """
    preds = torch.sigmoid(preds)
    mask = (targets != -1).float()
    preds = preds * mask
    targets = targets * mask

    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum()
    return 1 - (2. * intersection + smooth) / (union + smooth)

def focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    """
    Focal loss to address class imbalance, ignoring uncertain (-1) values.
    """
    targets = targets.float()
    mask = (targets != -1).float()
    logits = logits * mask
    targets = targets * mask

    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    pt = torch.exp(-bce)
    focal = alpha * (1 - pt) ** gamma * bce
    return (focal * mask).sum() / mask.sum().clamp(min=1.0)

def combo_loss(preds, targets, pos_weight=3.0, alpha=0.25, gamma=2.0,
               bce_weight=0.3, dice_weight=0.4, focal_weight=0.3):
    """
    Combines Weighted BCE, Dice Loss, and Focal Loss.
    """
    bce = weighted_bce_with_logits_masked(preds, targets, pos_weight)
    dsc = dice_loss(preds, targets)
    fcl = focal_loss(preds, targets, alpha=alpha, gamma=gamma)
    return bce_weight * bce + dice_weight * dsc + focal_weight * fcl



