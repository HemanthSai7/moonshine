import math
import tensorflow as tf

def get_num_batches(
    nsamples,
    batch_size,
    drop_remainders=True,
):
    if nsamples is None or batch_size is None:
        return None
    if drop_remainders:
        return math.floor(float(nsamples) / float(batch_size))
    return math.ceil(float(nsamples) / float(batch_size))

def get_conv_length(self, input_length):
    length = input_length
    
    for i in range(len(self.kernel_size)):
        if self.padding[i] == "same":
            length = tf.math.ceil(length / self.strides[i])
        elif self.padding[i] == "valid":
            length = tf.math.floor((length - self.kernel_size[i] + 1) / self.strides[i])
            
    return tf.cast(length, tf.int32)