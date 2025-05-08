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

def get_conv_length(input_length, kernel_size, padding, strides):
    length = input_length
    
    if padding == "same":
        length = tf.math.ceil(length / strides)
    elif padding == "valid":
        length = tf.math.floor((length - kernel_size + 1) / strides)
            
    return tf.cast(length, tf.int32)