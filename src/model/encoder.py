from src.utils import layer_util, math_util


import tensorflow as tf


class SubsamplingLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        kernel_size: list = [127, 7, 3],
        strides: list = [64 ,3, 2],
        padding: str = "causal",
        activation: list = ["tanh", "gelu", "gelu"],
    ):
        ...