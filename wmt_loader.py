# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 13:57:59 2019

@author: jbk48
"""


class Data(object):

    def __init__(self, path, max_enc_len=50, max_dec_len=50):

        self.path = path

        self.max_enc_len , self.max_dec_len = max_enc_len, max_dec_len
        
        self.pad_token, self.pad_idx = "<pad>", 0
        self.unk_token, self.unk_idx = "<unk>", 1
        self.bos_token, self.bos_idx = "<s>", 2
        self.eos_token, self.eos_idx = "</s>", 3

        self.w2idx = self.read_vocab()
        self.idx2w = dict(zip(self.w2idx.values(), self.w2idx.keys()))
        self.vocab = len(self.w2idx)
        

    def read_vocab(self):
        w2idx = {self.pad_token: self.pad_idx, self.unk_token: self.unk_idx, self.bos_token: self.bos_idx, self.eos_token: self.eos_idx}
        with open(self.path + "/" + "vocab.en", encoding="utf-8") as fin:
            lines = fin.readlines()
        for i, line in enumerate(lines):
            word = line.rstrip()
            if word not in w2idx:
                w2idx[word] = len(w2idx)
        with open(self.path + "/" + "vocab.vi", encoding="utf-8") as fin:
            lines = fin.readlines()
        for i, line in enumerate(lines):
            word = line.rstrip()
            if word not in w2idx:
                w2idx[word] = len(w2idx)
        return w2idx


    def read_file(self, name):
        print("preparing {} file".format(name))
        
        with open(self.path + "/" + name + ".vi", encoding="utf-8") as f:
            enc = f.readlines()
        
        with open(self.path + "/" + name + ".en", encoding="utf-8") as f:
            dec = f.readlines()
        
        enc_idx, enc_len = [], []
        dec_idx, dec_len = [], []
        
        for sent1, sent2 in zip(enc,dec):
            if(len(sent1.split(" ")) >self.max_enc_len-1):
                continue
            if(len(sent2.split(" ")) >self.max_dec_len-1):
                continue
            if(sent1 != "\n" and sent2 != "\n"):
                sent1 = sent1.replace("\n", " </s>").split(" ")
                enc_len.append(len(sent1))
                enc_idx.append(self.sent2idx(sent1, self.w2idx, self.max_enc_len))
            
                sent2 = sent2.replace("\n", " </s>").split(" ")
                dec_len.append(len(sent2))
                dec_idx.append(self.sent2idx(sent2, self.w2idx, self.max_enc_len))
        
        return enc_idx, dec_idx, enc_len, dec_len

    def sent2idx(self, sent, w2idx, max_len):
        idx = []
        for word in sent:
            if(word in w2idx):
                idx.append(w2idx[word])
            else:
                idx.append(self.unk_idx)
        
        return idx + [0]*(max_len-len(idx)) ## PAD for max length

