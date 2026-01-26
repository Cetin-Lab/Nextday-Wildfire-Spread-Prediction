# -*- coding: utf-8 -*-
"""
Created on Sat Jul 19 03:15:23 2025

@author: olivi
"""

import tensorflow as tf

def gaussian_kernel_tf(size: int, sigma: float) -> tf.Tensor:
    """Generates a 2D Gaussian kernel."""
    x = tf.range(-size // 2 + 1, size // 2 + 1, dtype=tf.float32)
    xx, yy = tf.meshgrid(x, x)
    kernel = tf.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    kernel = kernel / tf.reduce_sum(kernel)
    kernel = tf.expand_dims(tf.expand_dims(kernel, axis=-1), axis=-1)  # [k, k, 1, 1]
    return kernel

def tf_gaussian_prob_map(mask: tf.Tensor,
                         sigmas: list,
                         kernel_size: int = 9,
                         combine: str = "union") -> tf.Tensor:
    """TensorFlow version of gaussian_prob_map."""
    mask = tf.expand_dims(mask, axis=0)   # [1, H, W]
    mask = tf.expand_dims(mask, axis=-1)  # [1, H, W, 1]

    blurred_maps = []
    for sigma in sigmas:
        kernel = gaussian_kernel_tf(kernel_size, sigma)
        blurred = tf.nn.conv2d(mask, kernel, strides=1, padding="SAME")
        blurred_maps.append(blurred)

    stack = tf.concat(blurred_maps, axis=0)  # [N, H, W, 1]
    stack = tf.clip_by_value(stack, 0.0, 1.0)

    if combine == "mean":
        combined = tf.reduce_mean(stack, axis=0)
    elif combine == "max":
        combined = tf.reduce_max(stack, axis=0)
    elif combine == "union":
        combined = 1.0 - tf.reduce_prod(1.0 - stack, axis=0)
    else:
        raise ValueError("combine must be 'mean', 'max', or 'union'")

    return tf.squeeze(combined, axis=-1)  # [H, W, 1] → [H, W]

def tf_soften_prevfire(mask: tf.Tensor,
                       profile: str = "moderate",
                       combine: str = "union") -> tf.Tensor:
    """TensorFlow-native softening with pre-defined σ profiles."""
    SIGMA_PROFILES = {
        "fine":     [0.4, 0.8],
        "moderate": [0.7, 1.5, 3.0],
        "wide":     [1.0, 3.0, 6.0],
        "ultra":    [2.0, 4.0, 8.0, 12.0],
        "log5":     tf.exp(tf.linspace(tf.math.log(0.1), tf.math.log(5.0), 5)),
        "linear7":  tf.linspace(0.5, 7.0, 7),
    }

    if profile not in SIGMA_PROFILES:
        tf.print("[WARN] Unknown σ-profile:", profile)
        return tf.where(mask < 0, 0.0, mask)  # fallback

    sigmas = SIGMA_PROFILES[profile]
    return tf_gaussian_prob_map(mask, sigmas=sigmas, combine=combine)
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Assume these are already defined in scope
# from your_tf_gaussian_module import tf_soften_prevfire

# # Create toy binary mask (circle in center)
# H, W = 64, 64
# yy, xx = np.mgrid[:H, :W]
# binary_mask = ((yy - H//2)**2 + (xx - W//2)**2 <= 15**2).astype(np.float32)  # float32 for TensorFlow

# # Convert to tf.Tensor
# mask_tf = tf.convert_to_tensor(binary_mask)

# # Visualise
# SIGMA_PROFILES = {
#     "fine":     [0.4, 0.8],
#     "moderate": [0.7, 1.5, 3.0],
#     "wide":     [1.0, 3.0, 6.0],
#     "ultra":    [2.0, 4.0, 8.0, 12.0],
#     "log5":     tf.exp(tf.linspace(tf.math.log(0.1), tf.math.log(5.0), 5)),
#     "linear7":  tf.linspace(0.5, 7.0, 7),
# }

# fig, axs = plt.subplots(1, len(SIGMA_PROFILES)+1, figsize=(3*(len(SIGMA_PROFILES)+1), 3))

# axs[0].imshow(binary_mask, cmap="hot", vmin=0.0, vmax=0.5)
# axs[0].set_title("binary")
# axs[0].axis("off")

# for i, (name, sigmas) in enumerate(SIGMA_PROFILES.items(), start=1):
#     prob_map_tf = tf_soften_prevfire(mask_tf, profile=name)
#     prob_map_np = prob_map_tf.numpy()
#     axs[i].imshow(prob_map_np, cmap="hot", vmin=0.0, vmax=1.0)
#     axs[i].set_title(f"{name}\nσ={np.round(sigmas if isinstance(sigmas, list) else sigmas.numpy(), 2)}")
#     axs[i].axis("off")



# plt.tight_layout()
# plt.show()
