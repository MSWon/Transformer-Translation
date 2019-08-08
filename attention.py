# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 13:28:04 2019

@author: jbk48
"""

import tensorflow as tf


class Attention:
    """Attention class"""

    def __init__(self,
                 num_heads=6,
                 mode=None,
                 linear_key_dim=512,
                 linear_value_dim=512,
                 model_dim=512,
                 dropout=0.1):

        assert linear_key_dim % num_heads == 0
        assert linear_value_dim % num_heads == 0

        self.num_heads = num_heads
        self.mode = mode
        self.linear_key_dim = linear_key_dim
        self.linear_value_dim = linear_value_dim
        self.model_dim = model_dim
        self.dropout = dropout

    def multi_head(self, q, k, v, bias):
        q, k, v = self._linear_projection(q, k, v)
        qs, ks, vs = self._split_heads(q, k, v)
        outputs = self._scaled_dot_product(qs, ks, vs, bias)
        output = self._concat_heads(outputs)
        output = tf.layers.dense(output, self.model_dim, use_bias=False)
        return tf.nn.dropout(output, 1.0 - self.dropout)

    def _linear_projection(self, q, k, v):
        q = tf.layers.dense(q, self.linear_key_dim, use_bias=False, name="q")
        k = tf.layers.dense(k, self.linear_key_dim, use_bias=False, name="k")
        v = tf.layers.dense(v, self.linear_value_dim, use_bias=False, name="v")
        return q, k, v

    def _split_heads(self, q, k, v):        
        def split_last_dimension_then_transpose(tensor, num_heads, dim):
            t_shape = tensor.get_shape().as_list()
            length = t_shape[1]
            depth = (dim // num_heads)
            tensor = tf.reshape(tensor, [-1, length, num_heads, depth])
            return tf.transpose(tensor, [0, 2, 1, 3]) ## [batch_size, num_heads, length, depth]

        qs = split_last_dimension_then_transpose(q, self.num_heads, self.linear_key_dim)
        ks = split_last_dimension_then_transpose(k, self.num_heads, self.linear_key_dim)
        vs = split_last_dimension_then_transpose(v, self.num_heads, self.linear_value_dim)
        return qs, ks, vs

    def _scaled_dot_product(self, qs, ks, vs, bias):
        key_dim_per_head = self.linear_key_dim // self.num_heads

        o1 = tf.matmul(qs, ks, transpose_b=True) ## [batch_size, num_heads, query_len, key_len]
        logits = o1 / (key_dim_per_head**0.5) ## scaling
        logits += bias ## adding attention mask bias
        weights = tf.nn.softmax(logits, name="attention_weights")
        weights = tf.nn.dropout(weights, 1.0 - self.dropout)
        attention_output = tf.matmul(weights, vs)        
        return attention_output

    def _concat_heads(self, outputs):
        t_shape = outputs.get_shape().as_list()
        num_heads, length, depth = t_shape[1:]
        outputs = tf.transpose(outputs, [0, 2, 1, 3]) ## [batch, length, num_heads, depth]
        return tf.reshape(outputs, [-1, length, num_heads*depth])
