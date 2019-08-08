# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 13:57:59 2019

@author: jbk48
"""

from transformer_model import Model
import tensorflow as tf

if __name__ == '__main__':

    flags = tf.app.flags
    FLAGS = flags.FLAGS
    
    ## Model parameter
    flags.DEFINE_integer('hidden_dim', 512, 'dimension of hidden nodes')
    flags.DEFINE_integer('num_layers', 6, 'number of layers of transformer encoders')
    flags.DEFINE_integer('num_heads', 8, 'number of heads of transformer encoders')
    flags.DEFINE_integer('linear_key_dim', 512, 'dimension of key vector')
    flags.DEFINE_integer('linear_value_dim', 512, 'dimension of value vector')
    flags.DEFINE_integer('ffn_dim', 2048, 'dimension of feed forward network')
    flags.DEFINE_integer('enc_max_len', 100, 'encoder max length')
    flags.DEFINE_integer('dec_max_len', 100, 'decoder max length')
    flags.DEFINE_integer('batch_size', 64, 'number of batch size')
    flags.DEFINE_integer('warmup_steps', 4000, 'number warmup steps')
    flags.DEFINE_integer('training_epochs', 40, 'number of training epochs')
   
    print('========================')
    for key in FLAGS.__flags.keys():
        print('{} : {}'.format(key, getattr(FLAGS, key)))
    print('========================')
    ## Build model
    model = Model(FLAGS.hidden_dim, FLAGS.num_layers, FLAGS.num_heads, FLAGS.linear_key_dim, 
                  FLAGS.linear_value_dim, FLAGS.ffn_dim, FLAGS.enc_max_len, FLAGS.dec_max_len,
                  FLAGS.batch_size, FLAGS.warmup_steps)
    
    ## Train model
    model.train(FLAGS.training_epochs)
