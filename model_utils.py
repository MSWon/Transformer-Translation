# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 11:47:27 2019

@author: jbk48
"""

import math

import numpy as np
import tensorflow as tf

# Very low numbers to represent -infinity. We do not actually use -Inf, since we
# want to be able to multiply these values by zero to get zero. (-Inf * 0 = NaN)
_NEG_INF_FP32 = -1e9
_NEG_INF_FP16 = np.finfo(np.float16).min


def get_position_encoding(sentence_length, dim, dtype=tf.float32):
    pos_enc = np.array([[pos / np.power(10000., 2. * (i // 2) / dim) for i in range(dim)] for pos in range(sentence_length)]) # [seq_len, d_model]    
    # Apply the cosine to even columns and sin to odds.
    pos_enc[:, 0::2] = np.sin(pos_enc[:, 0::2])  # dim 2i
    pos_enc[:, 1::2] = np.cos(pos_enc[:, 1::2])  # dim 2i+1
    return tf.convert_to_tensor(pos_enc, dtype=dtype)


def get_decoder_self_attention_bias(length, dtype=tf.float32):
  """Calculate bias for decoder that maintains model's autoregressive property.
  Creates a tensor that masks out locations that correspond to illegal
  connections, so prediction at position i cannot draw information from future
  positions.
  Args:
    x: int tensor with shape [batch_size, length]
    length: int length of sequences in batch.
    dtype: The dtype of the return value.
  Returns:
    float tensor of shape [1, 1, length, length]
  """
  neg_inf = _NEG_INF_FP16 if dtype == tf.float16 else _NEG_INF_FP32
  with tf.name_scope("decoder_self_attention_bias"):
      valid_locs = tf.linalg.band_part(tf.ones([length, length], dtype=dtype),
                                     -1, 0)
      valid_locs = tf.reshape(valid_locs, [1, 1, length, length])
      decoder_bias = neg_inf * (1.0 - valid_locs)
  return decoder_bias


def get_padding(x, padding_value=0, dtype=tf.float32):
  """Return float tensor representing the padding values in x.
  Args:
    x: int tensor with any shape
    padding_value: int value that
    dtype: The dtype of the return value.
  Returns:
    float tensor with same shape as x containing values 0 or 1.
      0 -> non-padding, 1 -> padding
  """
  with tf.name_scope("padding"):
    return tf.cast(tf.equal(x, padding_value), dtype)


def get_padding_bias(x):
  """Calculate bias tensor from padding values in tensor.
  Bias tensor that is added to the pre-softmax multi-headed attention logits,
  which has shape [batch_size, num_heads, length, length]. The tensor is zero at
  non-padding locations, and -1e9 (negative infinity) at padding locations.
  Args:
    x: int tensor with shape [batch_size, length]
  Returns:
    Attention bias tensor of shape [batch_size, 1, 1, length].
  """
  with tf.name_scope("attention_bias"):
    padding = get_padding(x)
    attention_bias = padding * _NEG_INF_FP32
    attention_bias = tf.expand_dims(
        tf.expand_dims(attention_bias, axis=1), axis=1)
  return attention_bias


class LayerNormalization(tf.layers.Layer):
  """Applies layer normalization."""

  def __init__(self, hidden_size):
    super(LayerNormalization, self).__init__()
    self.hidden_size = hidden_size

  def build(self, _):
    self.scale = tf.get_variable("layer_norm_scale", [self.hidden_size],
                                 initializer=tf.ones_initializer())
    self.bias = tf.get_variable("layer_norm_bias", [self.hidden_size],
                                initializer=tf.zeros_initializer())
    self.built = True

  def call(self, x, epsilon=1e-6):
    mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
    variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keepdims=True)
    norm_x = (x - mean) * tf.rsqrt(variance + epsilon)
    return norm_x * self.scale + self.bias