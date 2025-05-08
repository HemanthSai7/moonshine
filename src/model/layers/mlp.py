from src.model.layers import get_activation
from typing import Callable, Union

import tensorflow as tf

@tf.keras.utils.register_keras_serializable(package=__name__)
class FFNModule(tf.keras.layers.Layer):
    def __init__(
        self,
        input_dim: int,
        dropout: float = 0.0,
        fc_factor: int = 4,
        activation: str = "gelu",
        kernel_regularizer: Union[str, Callable] = None,
        bias_regularizer: Union[str, Callable] = None,
        kernel_initializer: Union[str, Callable] = "glorot_uniform",
        bias_initializer: Union[str, Callable] = "zeros",
        name: str = "ffn_module",
        **kwargs,
    ):
        super(FFNModule, self).__init__(name=name, **kwargs)
        self.fc_factor = fc_factor
        self.ln = tf.keras.layers.LayerNormalization(
            name=f"{name}_ln",
            gamma_regularizer=kernel_regularizer,
            beta_regularizer=bias_regularizer,
        )
        self.ffn1 = tf.keras.layers.Dense(
            units=input_dim * fc_factor,
            activation="relu",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            name=f"{name}_ffn1",
        )
        self.act = get_activation(name=activation)
        self.do1 = tf.keras.layers.Dropout(rate=dropout, name=f"{name}_dropout1")
        self.ffn2 = tf.keras.layers.Dense(
            units=input_dim,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            name=f"{name}_ffn2",
        )
        self.do2 = tf.keras.layers.Dropout(rate=dropout, name=f"{name}_dropout2")
        self.res_add = tf.keras.layers.Add(name=f"{name}_res_add")

    def call(self, inputs, training=False):
        outputs = self.ffn1(inputs, training=training)
        outputs = self.act(outputs)
        outputs = self.do1(outputs, training=training)
        outputs = self.ffn2(outputs, training=training)
        outputs = self.do2(outputs, training=training)
        outputs = self.res_add([outputs, inputs])
        outputs = self.ln(outputs)
        return outputs
    
    def compute_output_shape(self, input_shape):
        # FFN preserves input shape
        return input_shape
    
    def get_config(self):
        conf = super().get_config()
        conf.update(self.ln.get_config())
        conf.update(self.ffn1.get_config())
        conf.update(self.act.get_config())
        conf.update(self.do1.get_config())
        conf.update(self.ffn2.get_config())
        conf.update(self.do2.get_config())
        return conf