# Transformer-Translation
## 1. Model
**Transformer model for translation** - Vi2En (Attention is all you need)
![alt_text](https://github.com/MSWon/Transformer-Translation/blob/master/images/model.png "Model")

## 2. Encoder mask
**Encoder**에서 <pad>에 관한 부분을 -inf로 masking을 함으로써 softmax시 0을 갖도록 한다.
![alt_text](https://github.com/MSWon/Transformer-Translation/blob/master/images/encoder_mask.png "Encoder mask")
```
def get_padding(x, padding_value=0, dtype=tf.float32):
  with tf.name_scope("padding"):
    return tf.cast(tf.equal(x, padding_value), dtype)
```
```
def get_padding_bias(x):
  with tf.name_scope("attention_bias"):
    padding = get_padding(x)
    attention_bias = padding * _NEG_INF_FP32
    attention_bias = tf.expand_dims(tf.expand_dims(attention_bias, axis=1), axis=1)
  return attention_bias
```

## 3. Decoder mask
**Decoder**에서는 미래를 볼 수 없기 때문에 미래에 대한 부분을 -inf로 masking을 함으로써 softmax시 0을 갖도록 한다.
![alt_text](https://github.com/MSWon/Transformer-Translation/blob/master/images/decoder_mask.png "Decoder mask")
```
def get_decoder_self_attention_bias(length, dtype=tf.float32):
  neg_inf = _NEG_INF_FP16 if dtype == tf.float16 else _NEG_INF_FP32
  with tf.name_scope("decoder_self_attention_bias"):
    valid_locs = tf.linalg.band_part(tf.ones([length, length], dtype=dtype), -1, 0)
    valid_locs = tf.reshape(valid_locs, [1, 1, length, length])
    decoder_bias = neg_inf * (1.0 - valid_locs)
  return decoder_bias
```
## 4. Feed foward network
**Encoder**부분에서 Feed foward network의 입력 부분에서 <pad>에 관한 부분에 대해서 **tf.gather_nd**를 통해 제거를 한 후, 출력 할 때 <pad> 부분에 0 vector를 채운다.
  
![alt_text](https://github.com/MSWon/Transformer-Translation/blob/master/images/feed_foward.png "Feed foward")
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
## 5. Results
2 layer, 8 heads에 관한 실험 결과

**1. Test loss**
![alt_text](https://github.com/MSWon/Transformer-Translation/blob/master/images/test_loss.png "Test loss")

**2. Test BLEU score**
![alt_text](https://github.com/MSWon/Transformer-Translation/blob/master/images/test_bleu.png "Test BLEU")
