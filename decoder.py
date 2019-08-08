import tensorflow as tf

from attention import Attention
from layer import FFN
from model_utils import LayerNormalization


class Decoder:
    """Decoder class"""

    def __init__(self,
                 num_layers=6,
                 num_heads=8,
                 linear_key_dim=512,
                 linear_value_dim=512,
                 model_dim=512,
                 ffn_dim=2048,
                 dropout=0.2):

        self.num_layers = num_layers
        self.num_heads = num_heads
        self.linear_key_dim = linear_key_dim
        self.linear_value_dim = linear_value_dim
        self.model_dim = model_dim
        self.ffn_dim = ffn_dim
        self.dropout = dropout
        self.layer_norm = LayerNormalization(self.model_dim)
        
    def build(self, decoder_inputs, encoder_outputs, dec_bias, enc_dec_bias):
        o1 = tf.identity(decoder_inputs) ## batch_size, dec_len, dim

        for i in range(1, self.num_layers+1):
            with tf.variable_scope("layer-{}".format(i)):
                o2 = self._add_and_norm(o1, self._masked_self_attention(q=o1,
                                                                        k=o1,
                                                                        v=o1,
                                                                        bias=dec_bias), num=1)
                o3 = self._add_and_norm(o2, self._encoder_decoder_attention(q=o2,
                                                                            k=encoder_outputs,
                                                                            v=encoder_outputs,
                                                                            bias=enc_dec_bias), num=2)
                o4 = self._add_and_norm(o3, self._positional_feed_forward(o3), num=3)
                o1 = tf.identity(o4)
        return o4

    def _masked_self_attention(self, q, k, v, bias):
        with tf.variable_scope("masked-self-attention"):
            attention = Attention(num_heads=self.num_heads,
                                    mode="masked-self-attention",  
                                    linear_key_dim=self.linear_key_dim,
                                    linear_value_dim=self.linear_value_dim,
                                    model_dim=self.model_dim,
                                    dropout=self.dropout)
            return attention.multi_head(q, k, v, bias)

    def _add_and_norm(self, x, sub_layer_x, num=0):
        with tf.variable_scope("add-and-norm-{}".format(num)):
            return self.layer_norm(tf.add(x, sub_layer_x)) # with Residual connection

    def _encoder_decoder_attention(self, q, k, v, bias):
        with tf.variable_scope("encoder-decoder-attention"):
            attention = Attention(num_heads=self.num_heads,
                                    mode="encoder-decoder-attention",
                                    linear_key_dim=self.linear_key_dim,
                                    linear_value_dim=self.linear_value_dim,
                                    model_dim=self.model_dim,
                                    dropout=self.dropout)
            return attention.multi_head(q, k, v, bias)

    def _positional_feed_forward(self, output):
        with tf.variable_scope("feed-forward"):
            ffn = FFN(w1_dim=self.ffn_dim,
                      w2_dim=self.model_dim,
                      dropout=self.dropout)
            return ffn.dense_relu_dense(output)


