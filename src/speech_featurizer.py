import tensorflow as tf

__all__ = [
    "SpeechFeaturizer",
]

@tf.keras.utils.register_keras_serializable(package=__name__)
class SpeechFeaturizer(tf.keras.layers.Layer):
    def __init__(
        self,
        sample_rate: int = 16000,
        feature_type: str = "waveform",
        normalize_signal: bool = False,
        preemphasis: float = 0.97,
        padding: float = 0.0,
        augmentation_config: dict = {},
        **kwargs,
    ):
        super().__init__(name=feature_type, **kwargs)
        self.sample_rate = sample_rate
        self.feature_type = feature_type
        self._normalize_signal = normalize_signal
        self.preemphasis = preemphasis
        self.padding = padding
        self.augmentation_config = augmentation_config

    def normalize_signal(self, signal: tf.Tensor) -> tf.Tensor:
        if self._normalize_signal:
            gain = 1.0 / (tf.reduce_max(tf.abs(signal), axis=-1) + 1e-9)
            return signal * gain
        return signal
    
    def preemphasis_signal(self, signal):
        if not self.preemphasis or self.preemphasis <= 0.0:
            return signal
        s0 = tf.expand_dims(signal[0], axis=-1)
        s1 = signal[1:] - self.preemphasis * signal[:-1]
        return tf.concat([s0, s1], -1)

    
    def call(self, inputs, training=False):
        signals, signals_length = inputs

        if training:
            signals, signals_length = self.augmentation.signal_augment(signals, signals_length)

        if self.padding > 0:
            signals = tf.pad(signals, [[0, 0], [0, self.padding]], mode="CONSTANT", constant_values=0.0)

        signals = self.normalize_signal(signals)
        signals = self.preemphasis_signal(signals)

        return signals, signals_length
    
    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1] + self.padding, 1
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "sample_rate": self.sample_rate,
            "feature_type": self.feature_type,
            "normalize_signal": self._normalize_signal,
            "preemphasis": self.preemphasis,
            "padding": self.padding,
            "augmentation_config": self.augmentation_config,
        })
        return config


