
import tensorflow as tf



class FFN:
    """FFN class (Position-wise Feed-Forward Networks)"""

    def __init__(self,
                 w1_dim=2048,
                 w2_dim=512,
                 dropout=0.1):

        self.w1_dim = w1_dim
        self.w2_dim = w2_dim
        self.dropout = dropout

    def dense_relu_dense(self, inputs, padding=None):
        
        """Return outputs of the feedforward network.
        
        Args:
          inputs: tensor with shape [batch_size, length, model_dim]
          padding: the padding values are temporarily removed
            from inputs. The padding values are placed
            back in the output tensor in the same locations.
            shape [batch_size, length]
            
        Returns:
          Output of the feedforward network.
          tensor with shape [batch_size, length, hidden_size]
        """           
        batch_size = tf.shape(inputs)[0]
        length = tf.shape(inputs)[1]
        '''
        t_shape = inputs.get_shape().as_list()
        length, model_dim = t_shape[1:]
        '''
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
                    output = tf.scatter_nd(
                        indices=nonpad_ids,
                        updates=output,
                        shape=[batch_size * length, self.w2_dim]
                    )
                    
                    output = tf.reshape(output, [batch_size, length, self.w2_dim])   
        
        return output

