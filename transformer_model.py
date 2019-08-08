# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 13:57:59 2019

@author: jbk48
"""

import os
import datetime
import tensorflow as tf
import numpy as np
import random
import pandas as pd
import model_utils

from wmt_loader import Data
from encoder import Encoder
from decoder import Decoder
from nltk.translate.bleu_score import corpus_bleu

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True


class Model(object):
    def __init__(self, emb_dim=512, num_layers=6, num_heads=8,
                 linear_key_dim=512, linear_value_dim=512, ffn_dim=2048,
                 max_enc_len=100, max_dec_len=100, batch_size=128, warmup_steps=4000):
        
        self.emb_dim = emb_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.linear_key_dim = linear_key_dim
        self.linear_value_dim = linear_value_dim
        self.ffn_dim = ffn_dim

        self.max_enc_len = max_enc_len
        self.max_dec_len = max_dec_len

        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        
        # Placeholder for Encoder
        self.x = tf.placeholder(dtype=tf.int32, shape=(None, max_enc_len))
        self.x_len = tf.placeholder(dtype=tf.int32, shape=(None, ))
        # Placeholder for Decoder
        self.y = tf.placeholder(dtype=tf.int32, shape=(None, max_dec_len))
        self.y_len = tf.placeholder(dtype=tf.int32, shape=(None, ))
        self.dropout = tf.placeholder(dtype=tf.float32, shape=())
        
        self.data = Data(path='./vi2en', max_enc_len=max_enc_len, max_dec_len=max_dec_len)
        self.bos_idx = self.data.bos_idx ## beginning of sentence
        self.eos_idx = self.data.eos_idx ## end of sentence
        self.vocab = self.data.vocab
        
        # Train
        self.train_enc, self.train_dec, self.train_enc_len, self.train_dec_len = self.data.read_file("train")
        # Val
        self.val_enc, self.val_dec, self.val_enc_len, self.val_dec_len = self.data.read_file("tst2012")
        # Test
        self.test_enc, self.test_dec, self.test_enc_len, self.test_dec_len = self.data.read_file("tst2013")
        print(' *---- Dataset Intialized ----\n')
        
        self.train_size = len(self.train_enc)
        self.test_size = len(self.test_enc)
        
        train_dataset = tf.data.Dataset.from_tensor_slices((self.x, self.x_len, self.y, self.y_len))
        train_dataset = train_dataset.shuffle(self.train_size)
        train_dataset = train_dataset.batch(self.batch_size)
        train_dataset = train_dataset.repeat()

        test_dataset = tf.data.Dataset.from_tensor_slices((self.x, self.x_len, self.y, self.y_len))
        test_dataset = test_dataset.batch(self.batch_size)
        test_dataset = test_dataset.repeat()
        
        iters = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
        self.iter_x, self.iter_x_len, self.iter_y, self.iter_y_len = iters.get_next()

        # create the initialisation operations
        self.train_init_op = iters.make_initializer(train_dataset)
        self.test_init_op = iters.make_initializer(test_dataset)
        print("building model")
        ## Output
        encoder_outputs, decoder_inputs, train_prob = self.decoder_train(self.iter_x, self.iter_y)
        self.build_loss(train_prob)
        self.global_step = tf.train.get_or_create_global_step()
        self.build_opt(self.global_step)
        
        self.pred_token = self.decoder_infer(self.iter_x)

        print("done")
        
    def train(self, training_epochs):
        
        num_train_batch = int(self.train_size / self.batch_size) + 1
        num_test_batch = int(self.test_size / self.batch_size) + 1
        
        train_feed_dict = {self.x: self.train_enc, self.x_len: self.train_enc_len,
                           self.y: self.train_dec, self.y_len: self.train_dec_len}     
        
        test_feed_dict = {self.x: self.test_enc, self.x_len: self.test_enc_len,
                          self.y: self.test_dec, self.y_len: self.test_dec_len}       
        
        print("vocab size : {}".format(self.vocab))
        print("start training")
        modelpath = "./model/"
        modelName = "transformer_de2en.ckpt"
        saver = tf.train.Saver()  
        
        with tf.Session(config=config) as sess:
            
            sess.run(tf.global_variables_initializer())

            if(not os.path.exists(modelpath)):
                os.mkdir(modelpath)
            ckpt = tf.train.get_checkpoint_state(modelpath)
            
            if(ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path)):
                saver.restore(sess, modelpath + modelName)
                print("Model loaded!")

            sess.run(self.train_init_op, feed_dict = train_feed_dict) 
            gs = sess.run(self.global_step)
                        
            start_time = datetime.datetime.now()
            
            train_loss_list = []
            test_loss_list = []
            test_bleu_list = []
            
            for epoch in range(training_epochs):
                
                train_loss = 0
                
                for step in range(num_train_batch):                         
                    _ = sess.run(self.clear_emb_W)
                    loss, _ = sess.run([self.loss, self.optimizer],
                                       feed_dict={self.dropout: 0.1})               
                    train_loss += loss/num_train_batch
                    print("epoch {:02d} step {:04d} loss {:.6f}".format(epoch+1, step+1, loss))
                                        
                print("Now for test data\nCould take few minutes")
                sess.run(self.test_init_op, feed_dict = test_feed_dict)
                test_loss = 0
                pred_list, true_list = [], []
                
                for step in range(num_test_batch):                  
                    loss = sess.run(self.loss, feed_dict={self.dropout: 0.0})
                    pred_, true_ = self.sample_test(self.data, sess)
                    pred_list += pred_
                    true_list += true_
                    test_loss += loss/num_test_batch
                
                bleu_score = corpus_bleu(true_list, pred_list)*100
                
                print("epoch {:02d} train loss {:.6f}".format(epoch+1, train_loss))   
                print("epoch {:02d} test loss {:.6f}".format(epoch+1, test_loss))
                print("epoch {:02d} bleu_score {:.6f}".format(epoch+1, bleu_score))
                train_loss_list.append(train_loss)
                test_loss_list.append(test_loss)
                test_bleu_list.append(bleu_score)
                sess.run(self.train_init_op, feed_dict = train_feed_dict) 
                save_path = saver.save(sess, modelpath + modelName)
                print ('save_path',save_path)
                
            result = pd.DataFrame({"train_loss":train_loss_list,
                                   "test_loss":test_loss_list,
                                   "test_bleu":test_bleu_list})
            
            result.to_csv("./loss.csv", sep =",", index=False)
            elapsed_time = datetime.datetime.now() - start_time
            print("{}".format(elapsed_time))
            

    def build_embed(self, inputs, encoder=True, reuse=False):
        with tf.variable_scope("Embeddings", reuse=reuse, initializer=tf.contrib.layers.xavier_initializer()):
            # Word Embedding
            self.emb_W = tf.get_variable('emb_W', [self.vocab, self.emb_dim], dtype = tf.float32)            
            self.clear_emb_W = tf.scatter_update(self.emb_W, [0], tf.constant(0.0, shape=[1, self.emb_dim]))
            embedding_inputs = self.emb_W
            
            if encoder:
                max_seq_length = self.max_enc_len
            else:
                max_seq_length = self.max_dec_len

            # Positional Encoding
            with tf.variable_scope("positional-encoding"):
                positional_encoded = model_utils.get_position_encoding(max_seq_length,
                                                                       self.emb_dim)
            batch_size = tf.shape(inputs)[0]

            ## Add
            word_emb = tf.nn.embedding_lookup(embedding_inputs, inputs)       
            position_inputs = tf.tile(tf.range(0, max_seq_length), [batch_size])
            position_inputs = tf.reshape(position_inputs, [batch_size, max_seq_length])
            position_emb = tf.nn.embedding_lookup(positional_encoded, position_inputs)
            position_emb = tf.where(tf.equal(word_emb, 0), word_emb, position_emb)                       
            encoded_inputs = tf.add(word_emb, position_emb)
            return tf.nn.dropout(encoded_inputs, 1.0 - self.dropout)

    def build_encoder(self, x, encoder_emb_inp, attention_bias, reuse=False):
        ## x: (batch_size, enc_len)
        padding_bias = attention_bias
        with tf.variable_scope("Encoder", reuse=reuse, initializer=tf.contrib.layers.xavier_initializer()):
            encoder = Encoder(num_layers=self.num_layers,
                              num_heads=self.num_heads,
                              linear_key_dim=self.linear_key_dim,
                              linear_value_dim=self.linear_value_dim,
                              model_dim=self.emb_dim,
                              ffn_dim=self.ffn_dim)
            ## padding = model_utils.get_padding(x)
            return encoder.build(encoder_emb_inp, padding_bias, padding=None)

    def build_decoder(self, decoder_emb_inp, encoder_outputs, dec_bias, attention_bias, reuse=False):
        enc_dec_bias = attention_bias
        with tf.variable_scope("Decoder", reuse=reuse, initializer=tf.contrib.layers.xavier_initializer()):
            decoder = Decoder(num_layers=self.num_layers,
                              num_heads=self.num_heads,
                              linear_key_dim=self.linear_key_dim,
                              linear_value_dim=self.linear_value_dim,
                              model_dim=self.emb_dim,
                              ffn_dim=self.ffn_dim)
            return decoder.build(decoder_emb_inp, encoder_outputs, dec_bias, enc_dec_bias)

    def build_output(self, decoder_outputs, reuse=False):
        with tf.variable_scope("Output", reuse=reuse):
            t_shape = decoder_outputs.get_shape().as_list() ## batch_size, seq_length, dim
            seq_length = t_shape[1]
            decoder_outputs = tf.reshape(decoder_outputs, [-1,self.emb_dim])
            logits = tf.matmul(decoder_outputs, self.emb_W, transpose_b=True)
            logits = tf.reshape(logits, [-1, seq_length, self.vocab])
        return logits
    
    def decoder_train(self, x, y):
        ## x: (batch_size, enc_len) , y: (batch_size, dec_len)
        dec_bias  = model_utils.get_decoder_self_attention_bias(self.max_dec_len)
        attention_bias = model_utils.get_padding_bias(x)
        # Encoder
        encoder_emb_inp = self.build_embed(x, encoder=True, reuse=False)
        encoder_outputs = self.build_encoder(x, encoder_emb_inp, attention_bias, reuse=False)
        # Decoder
        batch_size = tf.shape(x)[0]
        start_tokens = tf.fill([batch_size, 1], self.bos_idx) # 2: <s> ID
        target_slice_last_1 = tf.slice(y, [0, 0], [batch_size, self.max_dec_len-1])
        decoder_inputs = tf.concat([start_tokens, target_slice_last_1], axis=1) ## shift to right
        decoder_emb_inp = self.build_embed(decoder_inputs, encoder=False, reuse=True)
        decoder_outputs = self.build_decoder(decoder_emb_inp, encoder_outputs, dec_bias, attention_bias, reuse=False)
        train_prob = self.build_output(decoder_outputs, reuse=False)
        return encoder_outputs, decoder_inputs, train_prob

    def decoder_infer(self, x):   
        dec_bias  = model_utils.get_decoder_self_attention_bias(self.max_dec_len)
        attention_bias = model_utils.get_padding_bias(x)
        # Encoder
        encoder_emb_inp = self.build_embed(x, encoder=True, reuse=True)
        encoder_outputs = self.build_encoder(x, encoder_emb_inp, attention_bias, reuse=True)
        # Decoder
        batch_size = tf.shape(x)[0]
        start_tokens = tf.fill([batch_size, 1], self.bos_idx) # 2: <s> ID
        next_decoder_inputs = tf.concat([start_tokens, tf.zeros([batch_size, self.max_dec_len-1], dtype=tf.int32)], axis=1) ## batch_size, dec_len   
        # predict output with loop. [encoder_outputs, decoder_inputs (filled next token)]
        for i in range(1, self.max_dec_len):
            decoder_emb_inp = self.build_embed(next_decoder_inputs, encoder=False, reuse=True)
            decoder_outputs = self.build_decoder(decoder_emb_inp, encoder_outputs, dec_bias, attention_bias, reuse=True)
            logits = self.build_output(decoder_outputs, reuse=True)
            next_decoder_inputs = self._filled_next_token(next_decoder_inputs, logits, i)

        # slice start_token
        decoder_input_start_1 = tf.slice(next_decoder_inputs, [0, 1], [batch_size, self.max_dec_len-1])
        output_token = tf.concat([decoder_input_start_1, tf.zeros([batch_size, 1], dtype=tf.int32)], axis=1)
        return output_token

    def _filled_next_token(self, inputs, logits, decoder_index):
        batch_size = tf.shape(inputs)[0]
        next_token = tf.slice(
                tf.argmax(logits, axis=2, output_type=tf.int32),
                [0, decoder_index - 1],
                [batch_size, 1])
        left_zero_pads = tf.zeros([batch_size, decoder_index], dtype=tf.int32)
        right_zero_pads = tf.zeros([batch_size, (self.max_dec_len-decoder_index-1)], dtype=tf.int32)
        next_token = tf.concat((left_zero_pads, next_token, right_zero_pads), axis=1)
        return inputs + next_token

    def build_loss(self, train_prob):
        # sequence mask for different size
        self.masks = tf.sequence_mask(lengths=self.iter_y_len, maxlen=self.max_dec_len, dtype=tf.float32)
        y_ = self.label_smoothing(tf.one_hot(self.iter_y, depth=self.vocab))
        self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=train_prob)
        self.loss = tf.reduce_sum(self.cross_entropy * self.masks) / (tf.reduce_sum(self.masks) + 1e-10)

    def label_smoothing(self, inputs, epsilon=0.1):
        '''Applies label smoothing. See 5.4 and https://arxiv.org/abs/1512.00567.
        inputs: 3d tensor. [N, T, V], where V is the number of vocabulary.
        epsilon: Smoothing rate.
        '''
        V = inputs.get_shape().as_list()[-1] # number of channels
        return ((1-epsilon) * inputs) + (epsilon / V)
        
    def build_opt(self, global_step):
        # define optimizer
        learning_rate = self.noam_scheme(self.linear_key_dim , global_step, self.warmup_steps)
        self.optimizer = tf.contrib.opt.LazyAdamOptimizer(learning_rate=learning_rate, 
                                                          beta1=0.9, beta2=0.98, epsilon=1e-9).minimize(self.loss, global_step=global_step)
    def noam_scheme(self, d_model, global_step, warmup_steps=4000):
        '''Noam scheme learning rate decay
        init_lr: initial learning rate. scalar.
        global_step: scalar.
        warmup_steps: scalar. During warmup_steps, learning rate increases
            until it reaches init_lr.
        '''
        step = tf.cast(global_step + 1, dtype=tf.float32)
        return d_model ** (-0.5) * tf.minimum(step * warmup_steps ** -1.5, step ** -0.5)

    def restore(self, sess):
        print(' - Restoring variables...')
        var_list = [var for var in tf.all_variables()]
        saver = tf.train.Saver(var_list)
        saver.restore(sess, "models/model")
        print(' * model restored ')
        
    def sample_test(self, data, sess):        
        pred_token, enc, dec = sess.run([self.pred_token,self.iter_x, self.iter_y], feed_dict={self.dropout: 0.0})
        
        sample_idx = random.randint(0,len(pred_token)-1)
        pred_list = []
        true_list = []
        
        for i in range(len(pred_token)):            
            encoder = " ".join([data.idx2w[o] for o in enc[i]]).split("</s>")[0].strip()
            pred_line = " ".join([data.idx2w[o] for o in pred_token[i]]).split("</s>")[0].strip()
            true_line = " ".join([data.idx2w[o] for o in dec[i]]).split("</s>")[0].strip()           
            pred_list.append(pred_line.split(" "))
            true_list.append([true_line.split(" ")])
            
            if(i == sample_idx):
                sample_enc = encoder
                sample_pred = pred_line
                sample_true = true_line

        print("Encoder Input ===> {}".format(sample_enc).encode('utf-8'))
        print("Decoder True ===> {}".format(sample_true).encode('utf-8'))
        print("Decoder Pred ===> {}".format(sample_pred).encode('utf-8'))
        print("="*90)
        print()
        return pred_list, true_list