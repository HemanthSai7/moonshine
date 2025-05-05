from src.utils import math_util
from src.model.layers import MultiHeadAttention, RoPEPositionalEncoding, FFNModule

from typing import Callable, Union

import tensorflow as tf

EPSILON = 1e-6

__all__ = [
    "MoonshineEncoder",
]

@tf.keras.utils.register_keras_serializable(package=__name__)
class Conv1dSubsamplingLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        model_dim: int = 288,
        kernel_size: list = [127, 7, 3],
        strides: list = [64 ,3, 2],
        padding: str = ["same", "same", "same"],
        activations: list = ["tanh", "gelu", "gelu"],
        kernel_regularizer: str = None,
        bias_regularizer: str = None,
        kernel_initializer: str = "glorot_uniform",
        bias_initializer: str = "zeros",
        dropout_rate: float = 0.0,
        name: str = "conv1d_subsampling",
        **kwargs,
    ):
        super(Conv1dSubsamplingLayer, self).__init__(name=name, **kwargs)
        self.filters = [model_dim, 2 * model_dim, model_dim]
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding

        self.do = tf.keras.layers.Dropout(rate=dropout_rate, name=f"{name}_dropout")
        self.ln = tf.keras.layers.LayerNormalization(
                name=f"{name}_ln", gamma_regularizer=kernel_regularizer, beta_regularizer=bias_regularizer, dtype=tf.float32
        )

        self.conv = []
        for i in range(len(self.kernel_size)):
            conv = tf.keras.layers.Conv1D(
                filters=self.filters[i],
                kernel_size=self.kernel_size[i],
                strides=self.strides[i],
                padding=self.padding,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                name=f"{name}_conv_{i + 1}",
            )
            ac = tf.keras.layers.Activation(activations[i], name=f"{name}_activation_{i + 1}")
            self.conv.append({"conv": conv, "activation": ac})

    def call(self, inputs, training=False):
        inputs, inputs_length = inputs
        
        for i, conv in enumerate(self.conv):
            outputs = conv["conv"](inputs, training=training)
            outputs = conv["activation"](outputs)

            outputs_length = math_util.get_conv_length(
            inputs_length,
            self.kernel_size[i],
            self.strides[i],
            self.padding[i],
        )

        outputs = self.do(outputs, training=training)
        outputs = self.ln(outputs)

        outputs_length = tf.cast(outputs_length, dtype=tf.int32)
        return outputs, outputs_length
    
    def compute_output_shape(self, input_shape):
        inputs_length = input_shape[1]
        for i in range(len(self.kernel_size)):
            inputs_length = math_util.get_conv_length(
                inputs_length,
                self.kernel_size[i],
                self.strides[i],
                self.padding[i],
            )
        return (input_shape[0], inputs_length, self.filters[-1])
    
    def get_config(self):
        conf = super().get_config()
        conf.update({self.do.get_config()})
        conf.update({self.ln.get_config()})
        for conv in self.conv:
            conf.update(conv["conv"].get_config())
            conf.update(conv["activation"].get_config())
        return conf
    
@tf.keras.utils.register_keras_serializable(package=__name__)
class MHASModule(tf.keras.layers.Layer):
    def __init__(
        self,
        head_size: int,
        num_heads: int,
        dropout: float = 0.0,
        kernel_regularizer: Union[str, Callable] = None,
        bias_regularizer: Union[str, Callable] = None,
        kernel_initializer: Union[str, Callable] = "glorot_uniform",
        bias_initializer: Union[str, Callable] = "zeros",
        name: str = "mhas_module",
        **kwargs,
    ):
        super(MHASModule, self).__init__(name=name, **kwargs)
        self.ln = tf.keras.layers.LayerNormalization(
            name=f"{name}_ln",
            gamma_regularizer=kernel_regularizer,
            beta_regularizer=bias_regularizer,
        )
        self.mha = MultiHeadAttention(
            name=f"{name}_mha",
            head_size=head_size,
            num_heads=num_heads,
            dropout=dropout,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
        )
        self.do = tf.keras.layers.Dropout(dropout, name=f"{name}_dropout")
        self.res_add = tf.keras.layers.Add(name=f"{name}_res_add")

    def call(self, inputs, training=False):
        query, key, outputs = inputs
        outputs = self.mha([query, key, outputs], training=training)
        outputs = self.do(outputs, training=training)
        outputs = self.res_add([outputs, inputs])
        outputs = self.ln(outputs)
        return outputs
    
    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], input_shape[2]
    
    def get_config(self):
        conf = super().get_config()
        conf.update(self.ln.get_config())
        conf.update(self.mha.get_config())
        conf.update(self.do.get_config())
        conf.update(self.rotary_pos_emb.get_config())
        return conf

@tf.keras.utils.register_keras_serializable(package=__name__)
class MoonshineEncoderBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        input_dim: int,
        dropout: float = 0.0,
        fc_factor: int = 4,
        head_size: int = 64,
        num_heads: int = 4,
        kernel_regularizer: Union[str, Callable] = None,
        bias_regularizer: Union[str, Callable] = None,
        kernel_initializer: Union[str, Callable] = "glorot_uniform",
        bias_initializer: Union[str, Callable] = "zeros",
        name: str = "moonshine_encoder_block",
        **kwargs,
    ):
        super(MoonshineEncoderBlock, self).__init__(name=name, **kwargs)

        self.mhsa = MHASModule(
            head_size=head_size,
            num_heads=num_heads,
            dropout=dropout,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
        )
        self.ln = tf.keras.layers.LayerNormalization(
            name=f"{name}_ln",
            gamma_regularizer=kernel_regularizer,
            beta_regularizer=bias_regularizer,
        )
        self.ffn = FFNModule(
            input_dim=input_dim,
            dropout=dropout,
            fc_factor=fc_factor,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
        )
        self.ln = tf.keras.layers.LayerNormalization(
            name=f"{name}_ln",
            gamma_regularizer=kernel_regularizer,
            beta_regularizer=bias_regularizer,
        )

    def call(self, inputs, training=False, mask=None):
        query_pos_emb, key_pos_emb, outputs = inputs
        outputs = self.mhsa([query_pos_emb, key_pos_emb, outputs], training=training)
        outputs = self.ffn(outputs, training=training)
        outputs = self.ln(outputs)
        return outputs
    
    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], input_shape[2]
    
    def get_config(self):
        conf = super().get_config()
        conf.update(self.mhsa.get_config())
        conf.update(self.ln.get_config())
        conf.update(self.ffn.get_config())
        return conf

@tf.keras.utils.register_keras_serializable(package=__name__)
class MoonshineEncoder(tf.keras.Model):
    def __init__(
        self,
        input_dim: int,
        num_blocks: int = 6,
        dropout: float = 0.0,
        fc_factor: int = 4,
        head_size: int = 64,
        num_heads: int = 4,
        kernel_regularizer: Union[str, Callable] = None,
        bias_regularizer: Union[str, Callable] = None,
        kernel_initializer: Union[str, Callable] = "glorot_uniform",
        bias_initializer: Union[str, Callable] = "zeros",
        name: str = "moonshine_encoder",
        **kwargs,
    ):
        super(MoonshineEncoder, self).__init__(name=name, **kwargs)
        self.num_blocks = num_blocks
        self.conv_subsampling = Conv1dSubsamplingLayer(
            model_dim=input_dim,
            kernel_size=[127, 7, 3],
            strides=[64, 3, 2],
            padding=["same", "same", "same"],
            activations=["tanh", "gelu", "gelu"],
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
        )
        self.rotary_pos_emb = RoPEPositionalEncoding(name=f"{name}_rope_pos_emb")
        self.blocks = [
            MoonshineEncoderBlock(
                input_dim=input_dim,
                dropout=dropout,
                fc_factor=fc_factor,
                head_size=head_size,
                num_heads=num_heads,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
            )
            for _ in range(num_blocks)
        ]
        self.ln = tf.keras.layers.LayerNormalization(
            name=f"{name}_ln",
            gamma_regularizer=kernel_regularizer,
            beta_regularizer=bias_regularizer,
        )
        self.do = tf.keras.layers.Dropout(dropout, name=f"{name}_dropout")

    def call(self, inputs, training=False, mask=None):
        outputs, outputs_length = inputs
        outputs, outputs_length = self.conv_subsampling([outputs, outputs_length], training=training)
        outputs = self.do(outputs, training=training)

        query_pos_emb = self.rotary_pos_emb(outputs)
        key_pos_emb = self.rotary_pos_emb(outputs)

        for block in self.blocks:
            outputs = block([query_pos_emb, key_pos_emb, outputs], training=training, mask=mask)
        # outputs = self.ln(outputs)
        return outputs
    
    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], input_shape[2]
    
    def get_config(self):
        conf = super().get_config()
        conf.update(self.ln.get_config())
        conf.update(self.do.get_config())
        for block in self.blocks:
            conf.update(block.get_config())
        return conf
