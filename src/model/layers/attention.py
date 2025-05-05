from typing import Union, Callable

import tensorflow as tf

__all__ = [
    "MultiHeadAttention",
    "CausalMultiHeadAttention",
]

@tf.keras.utils.register_keras_serializable(package=__name__)
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        output_size: int = None,
        dropout: float = 0.0,
        return_attn_coef: bool = False,
        kernel_initializer: Union[str, Callable] = "glorot_uniform",
        kernel_regularizer: Union[str, Callable] = None,
        bias_regularizer: Union[str, Callable] = None,
        bias_initializer: Union[str, Callable] = "zeros",
        name: str = "multi_head_attention",
        **kwargs,
    ):
        super(MultiHeadAttention, self).__init__(name=name, **kwargs)
        self.num_heads = num_heads
        self.head_size = head_size
        self.output_size = output_size
        self.dropout = dropout
        self.return_attn_coef = return_attn_coef

        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.bias_initializer = bias_initializer

        self.dropout = tf.keras.layers.Dropout(dropout, name="dropout")
        self._do = dropout

    def build(self, input_shape):
        num_query_features = input_shape[0][-1]
        num_key_features = input_shape[1][-1]
        num_value_features = input_shape[2][-1] if len(input_shape) > 2 else num_key_features
        output_size = self.output_size if self.output_size is not None else num_key_features
        assert num_query_features == num_key_features, "Query and Key features must be the same"
        self.query_kernel = self.add_weight(
            name = "query_kernel",
            shape = [self.num_heads, num_query_features, self.head_size],
            initializer = self.kernel_initializer,
            regularizer = self.kernel_regularizer,
        )
        self.key_kernel = self.add_weight(
            name = "key_kernel",
            shape = [self.num_heads, num_key_features, self.head_size],
            initializer = self.kernel_initializer,
            regularizer = self.kernel_regularizer,
        )
        self.value_kernel = self.add_weight(
            name = "value_kernel",
            shape = [self.num_heads, num_value_features, self.head_size],
            initializer = self.kernel_initializer,
            regularizer = self.kernel_regularizer,
        )
        self.projection_kernel = self.add_weight(
            name="projection_kernel",
            shape = [self.num_heads, self.head_size, output_size],
            initializer = self.kernel_initializer,
            regularizer = self.kernel_regularizer,
        )

    def call_qkv(self, query, key, value):
        if key.shape[-2] != value.shape[-2]:
            raise ValueError("the number of elements in `key`  must be equal to " "the same as the number of elements in `value`")
        
        query = tf.einsum("...NI,HIO->...NHO", query, self.query_kernel)
        key = tf.einsum("...MI,HIO->...MHO", key, self.key_kernel)
        value = tf.einsum("...MI,HIO->...MHO", value, self.value_kernel)
        return query, key, value
    
    def call_attention(
        self,
        query,
        key,
        value,
        logits,
        training=False,
        mask=None,
    ):
        if mask is not None:
            if len(mask.shape) < 2:
                raise ValueError("`mask` must have at leat 2 dimensions")
            if query.shape[-3] != mask.shape[-2]:
                raise ValueError("masks's second to last dimension must be equal to " "the number of elements in `query`")
            if key.shape[-3] != mask.shape[-1]:
                raise ValueError("masks's last dimension must be equal to " "the number of elements in `key`")
            
        if mask is not None:
            mask = tf.cast(mask, tf.float32)

            if len(mask.shape) != len(logits.shape):
                mask = tf.expand_dims(mask, axis=-3)
            logits += -10e9 * (1.0 - mask)

        attn_coef = tf.nn.softmax(logits)
        attn_coef_dropout = self.dropout(attn_coef, training = training)
        multihead_output = tf.einsum("...HNM,...MHI->...NHI", attn_coef_dropout, value)
        output = tf.einsum("...NHI,HIO->...NO", multihead_output, self.projection_kernel)
        return output, attn_coef
    
    def call(self, inputs, training=False, mask=None, **kwargs):
        query, key, value = inputs

        query, key, value = self.call_qkv(query, key, value)
        depth = tf.constant(self.head_size, dtype=tf.float32)
        query /= tf.math.sqrt(depth)
        logits = tf.einsum("...NHO,...MHO->...HNM", query, key)
        output, attn_coef = self.call_attention(query, key, value, logits, training=training, mask=mask)

        if self.return_attn_coef:
            return output, attn_coef
        else:
            return output
        

@tf.keras.utils.register_keras_serializable(package=__name__)
class CausalMultiHeadAttention(MultiHeadAttention):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        output_size: int = None,
        dropout: float = 0.0,
        return_attn_coef: bool = False,
        kernel_initializer: Union[str, Callable] = "glorot_uniform",
        kernel_regularizer: Union[str, Callable] = None,
        bias_regularizer: Union[str, Callable] = None,
        bias_initializer: Union[str, Callable] = "zeros",
        name: str = "causal_multi_head_attention",
        **kwargs,
    ):
        super(CausalMultiHeadAttention, self).__init__(
            num_heads=num_heads,
            head_size=head_size,
            output_size=output_size,
            dropout=dropout,
            return_attn_coef=return_attn_coef,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            bias_initializer=bias_initializer,
            name=name,
            **kwargs
        )
        self._do = dropout

    def compute_causal_mask(self, query, value=None):
        q_seq_length = tf.shape(query)[1]
        v_seq_length = q_seq_length if value is None else tf.shape(value)[1]
        return tf.linalg.band_part(tf.ones((1, q_seq_length, v_seq_length), tf.bool), -1, 0)
    
    def call(self, inputs, training=False, mask=None, **kwargs):
        query, key, value = inputs

        query, key, value = self.call_qkv(query, key, value)
        depth = tf.constant(self.head_size, dtype=tf.float32)
        query /= tf.math.sqrt(depth)
        logits = tf.einsum("...NHO,...MHO->...HNM", query, key)

        causal_mask = self.compute_causal_mask(query, value)
        if mask is not None:
            causal_mask = tf.cast(causal_mask, tf.float32)
            mask = tf.cast(mask, tf.float32)
            mask = mask * causal_mask
        else:
            mask = causal_mask


        output, attn_coef = self.call_attention(query, key, value, logits, training=training, mask=mask)

        if self.return_attn_coef:
            return output, attn_coef
        else:
            return output
        
    def get_config(self):
        config = super(CausalMultiHeadAttention, self).get_config()
        config.update({
            "num_heads": self.num_heads,
            "head_size": self.head_size,
            "output_size": self.output_size,
            "dropout": self.dropout,
            "return_attn_coef": self.return_attn_coef,
            "kernel_initializer": self.kernel_initializer,
            "kernel_regularizer": self.kernel_regularizer,
            "bias_regularizer": self.bias_regularizer,
            "bias_initializer": self.bias_initializer,
        })
        return config