# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 13:57:59 2019

@author: jbk48
"""

from transformer_model import Model


emb_dim = 512          
num_layers = 6
num_heads = 8
linear_key_dim = 512
linear_value_dim = 512
ffn_dim = 2048
enc_max_len = 100     
dec_max_len = 100    

batch_size = 256       
warmup_steps = 4000
training_epochs = 30



model = Model(emb_dim=emb_dim, num_layers=num_layers, num_heads=num_heads,
              linear_key_dim=linear_key_dim, linear_value_dim=linear_value_dim,ffn_dim=ffn_dim,
              max_enc_len=enc_max_len, max_dec_len=dec_max_len,
              batch_size=batch_size, warmup_steps=warmup_steps)

model.train(training_epochs)
