# -*- coding: utf-8 -*-
"""
Created on Sat Sep 20 15:58:29 2025

@author: olivi
"""

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

"""Library of utility functions for reading TF Example datasets."""

import re
from typing import Dict, Text, Sequence, Optional, Tuple

import tensorflow as tf
import numpy as np

import constants
import image_utils
import prefire_gaussian  # Ensure this is in your PYTHONPATH


##################################
# New wrapper (replaces apply_soften_prevfire_tf)
def soften_prevfire_tf(mask: tf.Tensor,
                       profile: str = "moderate",
                       combine: str = "union") -> tf.Tensor:
    """TF wrapper around soften_prevfire (no wind scaling)."""
    def _soften(mask_np):
        softened = prefire_gaussian.soften_prevfire(
            np.asarray(mask_np, dtype=np.float32),
            profile=profile,
            combine=combine
        )
        return softened.astype(np.float32)

    softened = tf.py_function(_soften, [mask], tf.float32)
    softened.set_shape(mask.shape)  # preserve static shape
    return softened
##################################

class ModeKeys:
    TRAIN   = "train"
    EVAL    = "eval"
    PREDICT = "predict"
    
def get_features_dict(
    sample_size,
    features,
):
  """Creates a features dictionary for TensorFlow IO."""
  sample_shape = [sample_size, sample_size]
  features = set(features)
  columns = [tf.io.FixedLenFeature(shape=sample_shape, dtype=tf.float32)
            ] * len(features)
  return dict(zip(features, columns))


def map_fire_labels(labels):
  """Remaps the raw MODIS fire labels to fire, non-fire, and uncertain."""
  non_fire = tf.where(
      tf.logical_or(tf.equal(labels, 3), tf.equal(labels, 5)),
      tf.zeros_like(labels), -1 * tf.ones_like(labels))
  fire = tf.where(tf.greater_equal(labels, 7), tf.ones_like(labels), non_fire)
  return tf.cast(fire, dtype=tf.float32)


def get_num_channels(features, sequence_length = 1):
  if sequence_length == 1:
    return len(features)
  return len(features) // sequence_length


def _get_base_key(key):
  match = re.fullmatch(r'([a-zA-Z]+)', key)
  if match:
    return match.group(1)
  raise ValueError(
      f'The provided key does not match the expected pattern: {key}')


def _clip_and_rescale(inputs, key):
  base_key = _get_base_key(key)
  if base_key not in constants.DATA_STATS:
    raise ValueError(
        f'No data statistics available for the requested key: {key}.')
  min_val, max_val, _, _ = constants.DATA_STATS[base_key]
  inputs = tf.clip_by_value(inputs, min_val, max_val)
  return tf.math.divide_no_nan((inputs - min_val), (max_val - min_val))


def _clip_and_normalize(inputs, key):
  base_key = _get_base_key(key)
  if base_key not in constants.DATA_STATS:
    raise ValueError(
        f'No data statistics available for the requested key: {key}.')
  min_val, max_val, mean, std = constants.DATA_STATS[base_key]
  inputs = tf.clip_by_value(inputs, min_val, max_val)
  inputs = inputs - mean
  return tf.math.divide_no_nan(inputs, std)


def _validate_input_features(
    input_features = constants.INPUT_FEATURES):
  if not all(x in constants.INPUT_FEATURES for x in input_features):
    raise ValueError(f'input_features=[{input_features}] should be present in '
                     f'[{constants.INPUT_FEATURES}]')


def _validate_output_features(
    output_features = constants.OUTPUT_FEATURES):
  if not all(x in constants.OUTPUT_FEATURES for x in output_features):
    raise ValueError(
        f'output_features=[{output_features}] should be present in '
        f'[{constants.OUTPUT_FEATURES}]')


def adjust_prefire_mask(mask: tf.Tensor, apply_adjustment: bool = True) -> tf.Tensor:
    """Randomizes PrevFireMask values:
       - 0 → [0.01, 0.3]
       - 1 → [0.8, 0.99]
       - -1 → 0
    """
    if not apply_adjustment:
        return tf.where(mask < 0, 0.0, mask)

    mask = tf.where(mask < 0, 0.0, mask)

    rand_zeros = tf.random.uniform(shape=tf.shape(mask), minval=0.01, maxval=0.03)
    rand_ones  = tf.random.uniform(shape=tf.shape(mask), minval=0.8, maxval=0.99)

    mask = tf.where(mask == 1.0, rand_ones, mask)
    mask = tf.where(mask == 0.0, rand_zeros, mask)

    return mask


def _parse_journal2021_dataset(
    example_proto, input_sequence_length,
    output_sequence_length, data_size, input_features,
    output_features, clip_and_normalize,
    clip_and_rescale,
    gaussian_profile="moderate",
    gaussian_combine="union"):

  feature_names = list(input_features) + list(output_features)
  features_dict = get_features_dict(data_size, feature_names)
  features = tf.io.parse_single_example(example_proto, features_dict)

  prev_mask  = features.get('PrevFireMask')
  post_mask  = features.get('FireMask')

  prev_clean = tf.where(prev_mask  < 0, -1.0, prev_mask)
  post_clean = tf.where(post_mask  < 0, -1.0, post_mask)

  both_known = tf.logical_and(prev_clean >= 0, post_clean >= 0)
  combined   = tf.where(both_known,
                      tf.cast(tf.logical_or(prev_clean > 0, post_clean > 0),
                              tf.float32),
                      tf.constant(-1.0, dtype=tf.float32))

  if clip_and_normalize:
      inputs_list = []
      for key in input_features:
          value = _clip_and_normalize(features.get(key), key)

          if key == 'PrevFireMask':
              value = adjust_prefire_mask(value)
              value = soften_prevfire_tf(value,
                                         profile=gaussian_profile,
                                         combine=gaussian_combine)

          if key in ['vs', 'vd']:  # wind speed & wind direction maps
              value = soften_prevfire_tf(value,
                                         profile=gaussian_profile,
                                         combine=gaussian_combine)

          inputs_list.append(value)

  elif clip_and_rescale:
    inputs_list = [
        _clip_and_rescale(features.get(key), key) for key in input_features
    ]
  else:
    inputs_list = [features.get(key) for key in input_features]

  num_in_channels = get_num_channels(input_features, input_sequence_length)
  inputs_stacked = tf.stack(inputs_list, axis=0)
  if input_sequence_length > 1:
    inputs_stacked = tf.reshape(inputs_stacked,
                                [-1, num_in_channels, data_size, data_size])
    input_img = tf.transpose(inputs_stacked, [0, 2, 3, 1])
  else:
    input_img = tf.transpose(inputs_stacked, [1, 2, 0])
    
  outputs_list = [combined]

  if not outputs_list:
    raise ValueError('outputs_list should not be empty.')
  outputs_stacked = tf.stack(outputs_list, axis=0)
  if output_sequence_length > 1:
    outputs_stacked = tf.reshape(outputs_stacked, [-1, 1, data_size, data_size])
    output_img = tf.transpose(outputs_stacked, [0, 2, 3, 1])
  else:
    output_img = tf.transpose(outputs_stacked, [1, 2, 0])
  return input_img, output_img


def _parse_fn(
    example_proto, input_sequence_length,
    output_sequence_length, data_size, sample_size,
    output_sample_size, downsample_threshold, binarize_output,
    input_features, output_features,
    clip_and_normalize, clip_and_rescale, random_flip,
    random_rotate, random_crop, center_crop,
    azimuth_in_channel,
    azimuth_out_channel,
    gaussian_profile,
    gaussian_combine):
  if random_crop and center_crop:
    raise ValueError('Cannot have both random_crop and center_crop be True')

  input_img, output_img = _parse_journal2021_dataset(
      example_proto, input_sequence_length, output_sequence_length, data_size,
      input_features, output_features, clip_and_normalize, clip_and_rescale,
      gaussian_profile, gaussian_combine)

  num_in_channels = get_num_channels(input_features, input_sequence_length)
  num_out_channels = get_num_channels(output_features, output_sequence_length)

  if random_flip:
    input_img, output_img = image_utils.random_flip_input_and_output_images(
        input_img, output_img, azimuth_in_channel, azimuth_out_channel)
  if random_rotate:
    input_img, output_img = image_utils.random_rotate90_input_and_output_images(
        input_img, output_img, azimuth_in_channel, azimuth_out_channel)
  if random_crop:
    input_img, output_img = image_utils.random_crop_input_and_output_images(
        input_img, output_img, sample_size, num_in_channels, num_out_channels)
  if center_crop:
    input_img, output_img = image_utils.center_crop_input_and_output_images(
        input_img, output_img, sample_size)
  output_img = image_utils.downsample_output_image(output_img,
                                                   output_sample_size,
                                                   downsample_threshold,
                                                   binarize_output)
  return input_img, output_img


def get_dataset(file_pattern,
                data_size,
                sample_size,
                output_sample_size,
                batch_size,
                input_features,
                output_features,
                shuffle,
                shuffle_buffer_size,
                compression_type,
                input_sequence_length,
                output_sequence_length,
                repeat,
                clip_and_normalize,
                clip_and_rescale,
                random_flip,
                random_rotate,
                random_crop,
                center_crop,
                azimuth_in_channel,
                azimuth_out_channel,
                downsample_threshold = 0.0,
                binarize_output = True,
                gaussian_profile="moderate",
                gaussian_combine="union"):
  if (clip_and_normalize and clip_and_rescale):
    raise ValueError('Cannot have both normalize and rescale.')
  dataset = tf.data.Dataset.list_files(file_pattern, shuffle=shuffle)
  dataset = dataset.interleave(
      lambda x: tf.data.TFRecordDataset(x, compression_type=compression_type),
      num_parallel_calls=tf.data.experimental.AUTOTUNE)
  dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
  dataset = dataset.map(
      lambda x: _parse_fn(
          x, input_sequence_length, output_sequence_length, data_size,
          sample_size, output_sample_size, downsample_threshold,
          binarize_output, input_features, output_features, clip_and_normalize,
          clip_and_rescale, random_flip, random_rotate, random_crop,
          center_crop, azimuth_in_channel, azimuth_out_channel,
          gaussian_profile, gaussian_combine),
      num_parallel_calls=tf.data.experimental.AUTOTUNE)

  if shuffle:
    dataset = dataset.shuffle(shuffle_buffer_size)
  dataset = dataset.batch(batch_size)
  if repeat:
    dataset = dataset.repeat()
  dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
  return dataset


def make_dataset(
    hparams,
    mode = ModeKeys.TRAIN
):
  input_features = list(hparams.input_features)
  output_features = list(hparams.output_features)

  _validate_input_features(input_features)
  _validate_output_features(output_features)

  if (not hparams.azimuth_in_channel or
      hparams.azimuth_in_channel not in input_features):
    azimuth_in_channel = None
  else:
    azimuth_in_channel = input_features.index(hparams.azimuth_in_channel)

  if (not hparams.azimuth_out_channel or
      hparams.azimuth_out_channel not in output_features):
    azimuth_out_channel = None
  else:
    azimuth_out_channel = output_features.index(hparams.azimuth_out_channel)

  if mode == ModeKeys.TRAIN:
    file_pattern = hparams.train_path
    shuffle = True
    repeat = True
    random_flip = hparams.random_flip
    random_rotate = hparams.random_rotate
    random_crop = hparams.random_crop
    center_crop = False
  elif mode == ModeKeys.EVAL:
    file_pattern = hparams.eval_path
    shuffle = True
    repeat = True
    random_flip = False
    random_rotate = False
    random_crop = hparams.random_crop
    center_crop = False
  elif mode == ModeKeys.PREDICT:
    file_pattern = hparams.test_path  #############for interfere
    shuffle = False
    repeat = False
    random_flip = False
    random_rotate = False
    random_crop = hparams.random_crop
    center_crop = False
  else:
    raise NotImplementedError(f'Unsupported mode {mode}.')

  return get_dataset(
      file_pattern,
      data_size=hparams.data_sample_size,
      sample_size=hparams.sample_size,
      output_sample_size=hparams.output_sample_size,
      batch_size=hparams.batch_size,
      input_features=hparams.input_features,
      output_features=hparams.output_features,
      shuffle=shuffle,
      shuffle_buffer_size=hparams.shuffle_buffer_size,
      compression_type=hparams.compression_type,
      input_sequence_length=hparams.input_sequence_length,
      output_sequence_length=hparams.output_sequence_length,
      repeat=repeat,
      clip_and_normalize=True,
      clip_and_rescale=False,
      random_flip=random_flip,
      random_rotate=random_rotate,
      random_crop=random_crop,
      center_crop=center_crop,
      azimuth_in_channel=azimuth_in_channel,
      azimuth_out_channel=azimuth_out_channel,
      downsample_threshold=hparams.downsample_threshold,
      binarize_output=hparams.binarize_output,
      gaussian_profile=hparams.gaussian_profile,
      gaussian_combine=hparams.gaussian_combine)
