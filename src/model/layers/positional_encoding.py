from src.utils.shape_util import shape_list

import tensorflow as tf

__all__ = [
    "RoPEPositionalEncoding",
]

@tf.keras.utils.register_keras_serializable(package=__name__)
class RoPEPositionalEncoding(tf.keras.layers.Layer):
    def __init__(
        self,
        name: str = "rope_positional_encoding",
        **kwargs,
    ):
        super(RoPEPositionalEncoding, self).__init__(name=name, **kwargs)

    def build(
        self,
        input_shape,
    ):
        dmodel = input_shape[-1]
        assert dmodel % 2 == 0, "dmodel must be even"

    @staticmethod
    def encode(
        max_len: int,
        dmodel: int,
    ):
        pos = tf.expand_dims(tf.range(0, max_len, dtype=tf.float32), axis=1)

        dim = dmodel // 2
        index = tf.expand_dims(tf.range(0, dim, dtype=tf.float32), axis=0)
        freq = 1.0 / tf.pow(10000.0, (index / dmodel))

        theta = tf.matmul(pos, freq)

        sin = tf.sin(theta)
        cos = tf.cos(theta)

        return cos, sin
    
    def apply_rotary(self, x, cos, sin):
        batch, seq_len, dim = shape_list(x)
        half_dim = dim // 2

        x_reshape = tf.reshape(x, [batch, seq_len, half_dim, 2])

        cos = tf.squeeze(cos, axis=0)
        sin = tf.squeeze(sin, axis=0)

        rotation_matrix_00 = cos
        rotation_matrix_01 = -sin
        rotation_matrix_10 = sin
        rotation_matrix_11 = cos

        rotation_matrix = tf.stack(
            [rotation_matrix_00, rotation_matrix_01,
             rotation_matrix_10, rotation_matrix_11], 
             axis=-1
        )
        rotation_matrix = tf.reshape(rotation_matrix, [seq_len, half_dim, 2, 2])

        x_rotated = tf.einsum("bsdi,sdij->bsdj", x_reshape, rotation_matrix)
        x_out = tf.reshape(x_rotated, [batch, seq_len, dim])
        return x_out
    
    def call(self, inputs, **kwargs):
        batch, seq_len, dim = shape_list(inputs)
        cos, sin = self.encode(seq_len, dim)

        cos = tf.expand_dims(cos, axis=0)
        sin = tf.expand_dims(sin, axis=0)

        cos = tf.cast(cos, dtype=inputs.dtype)
        sin = tf.cast(sin, dtype=inputs.dtype)

        return self.apply_rotary(inputs, cos, sin)
    
    def get_config(self):
        config = super(RoPEPositionalEncoding, self).get_config()
        return config


        