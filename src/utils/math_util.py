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