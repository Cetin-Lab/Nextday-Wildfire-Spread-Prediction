import re
from typing import Dict, List, Optional, Text, Tuple
import matplotlib.pyplot as plt
from matplotlib import colors

import tensorflow as tf
import numpy as np


from tqdm import tqdm
from typing import Callable, Tuple

from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

from tensorflow.keras import backend as K
from tensorflow.python.keras.utils.losses_utils import reduce_weighted_loss


from dataset import make_dataset, ModeKeys
from constants import INPUT_FEATURES
# Let's load training, validation and test sets using function we define above

BATCH_SIZE = 32


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
    
    
    use_prefire_noise    = True   # calls adjust_prefire_mask()
    use_prefire_gaussian = False  # calls apply_soften_prevfire_tf()
    combine_fire_masks   = True   # PrevFireMask ⊕ FireMask
    
    
    # Gaussian softening (defaults; will be overwritten in the sweep)
    gaussian_profile = "moderate"
    gaussian_combine = "union"
    
    
train_ds = make_dataset(HParams, mode=ModeKeys.TRAIN)
# # Visualize data
# We will check content of the dataset by plotting them

# Let's plot the data!
# 
# First we define the names for each of our variables.

TITLES = [
  'Elevation',
  'Wind\ndirection',
  'Wind\nvelocity',
  'Min\ntemp',
  'Max\ntemp',
  'Humidity',
  'Precip',
  'Drought',
  'Vegetation',
  'Population\ndensity',
  'Energy\nrelease\ncomponent',
  'Prev-fire\nmask',
  'Post-fire\nmask'
]


# Define some helper variables for the plot. 
import os

def plot_samples_from_dataset(dataset: tf.data.Dataset,
                              n_rows: int,
                              save_path: Optional[str] = None,
                              dpi: int = 300,
                              show: bool = True):
    """
    Plot `n_rows` samples from a dataset and optionally save to disk.

    Args:
        dataset (tf.data.Dataset): Dataset to visualize
        n_rows (int): Number of rows to plot
        save_path (str, optional): Path to save the figure (PNG/PDF/etc.)
        dpi (int): Resolution for saved figure
        show (bool): Whether to display the figure
    """
    global TITLES

    # Get one batch
    for inputs, labels in dataset.take(1):
        pass

    fig = plt.figure(figsize=(15,3.5))

    # Fire mask colormap
    CMAP = colors.ListedColormap(['black', 'silver', 'orangered'])
    BOUNDS = [-1, -0.1, 0.001, 1]
    NORM = colors.BoundaryNorm(BOUNDS, CMAP.N)

    n_features = 12  # number of input channels visualized

    for i in range(n_rows):
        for j in range(n_features + 1):
            ax = plt.subplot(n_rows, n_features + 1,
                             i * (n_features + 1) + j + 1)

            if i == 0:
                ax.set_title(TITLES[j], fontsize=13)

            if j < n_features - 1:
                ax.imshow(inputs[i, :, :, j], cmap="viridis")
            elif j == n_features - 1:
                ax.imshow(inputs[i, :, :, -1], cmap=CMAP, norm=NORM)
            else:
                ax.imshow(labels[i, :, :, 0], cmap=CMAP, norm=NORM)

            ax.axis("off")


    plt.tight_layout(pad=0.2)

    plt.subplots_adjust(
        left=0.02,
        right=0.98,
        top=0.95,
        bottom=0.05,
        wspace=0.05,
        hspace=0.00
    )

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"🖼️ Figure saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


plot_samples_from_dataset(
    train_ds,
    n_rows=2,
    save_path="D:/wildfire/paper/dataset_examples/train_samples.png",
    dpi=300,
    show=True
)
