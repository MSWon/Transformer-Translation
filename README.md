# Transformer-Translation
## 1. Description
1. **Transformer model for translation** ([Attention is all you need](https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf))
2. This code was created by referring to the code in [DongjunLee](https://github.com/DongjunLee/transformer-tensorflow), [Kyubyong](https://github.com/Kyubyong/transformer), [Official code](https://github.com/tensorflow/models/tree/master/official/transformer)
3. Trained on 1 gpu(Geforce Gtx 1080 ti)

![alt_text](https://github.com/MSWon/Transformer-Translation/blob/master/images/model.png "Model")

## 2. Data
[Vietnamese to English data (IWSLT'15)](https://nlp.stanford.edu/projects/nmt/)

## 3. Train
**1. Git clone**
```
$ git clone https://github.com/MSWon/Transformer-Translation.git
```
**2. Training with user settings**
```
$ python train_transformer.py --num_layers 4 --num_heads 8 --batch_size 64 --training_epochs 25
```

# Training tips
1. **Transformer**을 학습을 할 때 중요한 것은 max_length로 padding을 한 **PAD**에 대해서 처리를 해주어야 한다.
2. 이를 위해 **Transformer**에서는 **Encoder, Decoder, Encoder-Decoder attention**에 대해서 적절히 **masking**을 한다.

## 1. Encoder mask
**Encoder**에서 **PAD**에 관한 부분을 **-inf**로 **masking**을 함으로써 softmax시 0을 갖도록 한다.
![alt_text](https://github.com/MSWon/Transformer-Translation/blob/master/images/encoder_mask.png "Encoder mask")

[model_utils.py](https://github.com/MSWon/Transformer-Translation/blob/master/model_utils.py#L48)
```
def get_padding(x, padding_value=0, dtype=tf.float32):
  with tf.name_scope("padding"):
    return tf.cast(tf.equal(x, padding_value), dtype)
```

[model_utils.py](https://github.com/MSWon/Transformer-Translation/blob/master/model_utils.py#L62)
```
def get_padding_bias(x):
  with tf.name_scope("attention_bias"):
    padding = get_padding(x)
    attention_bias = padding * _NEG_INF_FP32
    attention_bias = tf.expand_dims(tf.expand_dims(attention_bias, axis=1), axis=1)
  return attention_bias
```

## 2. Decoder mask
**Decoder**에서는 미래를 볼 수 없기 때문에 미래에 대한 부분을 **-inf**로 **masking**을 함으로써 softmax시 0을 갖도록 한다.
![alt_text](https://github.com/MSWon/Transformer-Translation/blob/master/images/decoder_mask.png "Decoder mask")

[model_utils.py](https://github.com/MSWon/Transformer-Translation/blob/master/model_utils.py#L27)
```
def get_decoder_self_attention_bias(length, dtype=tf.float32):
  neg_inf = _NEG_INF_FP16 if dtype == tf.float16 else _NEG_INF_FP32
  with tf.name_scope("decoder_self_attention_bias"):
    valid_locs = tf.linalg.band_part(tf.ones([length, length], dtype=dtype), -1, 0)
    valid_locs = tf.reshape(valid_locs, [1, 1, length, length])
    decoder_bias = neg_inf * (1.0 - valid_locs)
  return decoder_bias
```

## 3. Encoder-Decoder attention mask
**Encoder-Decoder attention**시에 **key, value vector**에 해당하는 **PAD**부분에 대해서 **-inf masking**을 해준다.
![alt_text](https://github.com/MSWon/Transformer-Translation/blob/master/images/encoder_decoder_mask.png "Encoder Deecoder mask")

[model_utils.py](https://github.com/MSWon/Transformer-Translation/blob/master/model_utils.py#L48)
```
def get_padding_bias(x):
  with tf.name_scope("attention_bias"):
    padding = get_padding(x)
    attention_bias = padding * _NEG_INF_FP32
    attention_bias = tf.expand_dims(tf.expand_dims(attention_bias, axis=1), axis=1)
  return attention_bias
```

## 4. Feed foward network
1. **Encoder**부분에서 Feed foward network의 입력 부분에서 **PAD**에 관한 부분에 대해서 **tf.gather_nd**를 통해 제거 
2. **tf.scatter_nd**를 통해 출력 할 때 **PAD** 부분에 **0 vector**를 채운다.
  
![alt_text](https://github.com/MSWon/Transformer-Translation/blob/master/images/feed_foward.png "Feed foward")

[layer.py](https://github.com/MSWon/Transformer-Translation/blob/master/layer.py#L18)
```
def dense_relu_dense(self, inputs, padding=None):
        if padding is not None:
            with tf.name_scope("remove_padding"):
                # Flatten padding to [batch_size*length]
                pad_mask = tf.reshape(padding, [-1])        
                nonpad_ids = tf.to_int32(tf.where(pad_mask < 1e-9))       
                # Reshape inputs to [batch_size*length, hidden_size] to remove padding
                inputs = tf.reshape(inputs, [-1, self.w2_dim])
                inputs = tf.gather_nd(inputs, indices=nonpad_ids)        
                # Reshape inputs from 2 dimensions to 3 dimensions.
                inputs = tf.expand_dims(inputs, axis=0)
            
        output = tf.layers.dense(inputs, self.w1_dim, activation=tf.nn.relu)
        output = tf.nn.dropout(output, 1.0 - self.dropout)
        output = tf.layers.dense(output, self.w2_dim)
        if padding is not None:
            with tf.name_scope("re_add_padding"):
                    output = tf.squeeze(output, axis=0)
                    output = tf.scatter_nd(indices=nonpad_ids, updates=output, shape=[batch_size * length, self.w2_dim])
                    output = tf.reshape(output, [batch_size, length, self.w2_dim])           
        return output
```
## 5. Shared embedding matrix
**Transformer**에서는 **Encoder**, **Decoder**, **Output softmax layer**에서 같은 **embedding matrix**를 사용한다.

![alt_text](https://github.com/MSWon/Transformer-Translation/blob/master/images/shared_embedding.png "Shared embedding")

[transformer_model.py](https://github.com/MSWon/Transformer-Translation/blob/master/transformer_model.py#L175)
```
def build_embed(self, inputs, encoder=True, reuse=False):
    with tf.variable_scope("Embeddings", reuse=reuse, initializer=tf.contrib.layers.xavier_initializer()):
        # Word Embedding
        self.shared_weights = tf.get_variable('shared_weights', [self.vocab, self.emb_dim], dtype = tf.float32)            

        if encoder:
            max_seq_length = self.max_enc_len
        else:
            max_seq_length = self.max_dec_len

        # Positional Encoding
        with tf.variable_scope("positional-encoding"):
            positional_encoded = model_utils.get_position_encoding(max_seq_length,
                                                                   self.emb_dim)
        batch_size = tf.shape(inputs)[0]
        mask = tf.to_float(tf.not_equal(inputs, 0))
        ## Add
        word_emb = tf.nn.embedding_lookup(self.shared_weights, inputs)   ## batch_size, length, dim
        word_emb *= tf.expand_dims(mask, -1) ## zeros out masked positions
        word_emb *= self.emb_dim ** 0.5 ## Scale embedding by the sqrt of the hidden size
        position_inputs = tf.tile(tf.range(0, max_seq_length), [batch_size])
        position_inputs = tf.reshape(position_inputs, [batch_size, max_seq_length])
        position_emb = tf.nn.embedding_lookup(positional_encoded, position_inputs)                       
        encoded_inputs = tf.add(word_emb, position_emb)
        return tf.nn.dropout(encoded_inputs, 1.0 - self.dropout)
```
## 6. Learning rate
**Transformer**에서는 warmup_step(4000)까지 linear하게 learning rate를 증가시키고, 이후에는 감소시킨다.

![alt_text](https://github.com/MSWon/Transformer-Translation/blob/master/images/learning_rate.png "learning rate")

[transformer_model.py](https://github.com/MSWon/Transformer-Translation/blob/master/transformer_model.py#L304)
```
def noam_scheme(self, d_model, global_step, warmup_steps=4000):
    step = tf.cast(global_step + 1, dtype=tf.float32)
    return d_model ** (-0.5) * tf.minimum(step * warmup_steps ** -1.5, step ** -0.5)
```

## 7. Results

1. 4 layer, 8 heads, 512 hidden dimension 에 관한 실험 결과 (**BLEU : 24.74**)
2. 6 layer, 8 heads, 512 hidden dimension, 16000 warmup steps 에 관한 실험 결과 (**BLEU : 26.42**)
3. Beam search에 관한 것은 구현이 안된 상태이다.

**1. Test loss**

![alt_text](https://github.com/MSWon/Transformer-Translation/blob/master/images/test_loss.png "Test loss")

**2. Test BLEU score**

![alt_text](https://github.com/MSWon/Transformer-Translation/blob/master/images/test_bleu.png "Test BLEU")

**3. Example**
```
Encoder Input ===> Tôi đã được lớn lên ở một quốc gia đã bị tiêu huỷ bởi bao thập niên chiến tranh .
Decoder True ===> I was raised in a country that has been destroyed by decades of war .
Decoder Pred ===> I grew up in a country that was <unk> by the war .
==========================================================================================
Encoder Input ===> Họ đều là nạn nhân của tổn thương , bệnh tật và bạo lực .
Decoder True ===> All of them are victim to injury , illness and violence .
Decoder Pred ===> They were all the victims of vulnerability , disease and violence .
==========================================================================================
Encoder Input ===> Và sau một vài tháng , tôi nhận ra rằng em khác biệt .
Decoder True ===> And after a few months went by , I realized that he was different .
Decoder Pred ===> And after a few months , I realized that you were different .
```
