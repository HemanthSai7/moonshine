from src.utils import file_util, data_util
from src.schemas import TrainInput

import tensorflow as tf


class BaseModel(tf.keras.Model):
    def __init__(self, **kwargs):
        super(BaseModel, self).__init__(**kwargs)
    
    def save(
        self,
        filepath: str,
        overwrite: bool = True,
        include_optimizer: bool = True,
        save_format: str = None,
        signatures: dict = None,
        options: tf.saved_model.SaveOptions = None,
        save_traces: bool = True,
    ):
        with file_util.save_file(filepath) as path:
            super(BaseModel, self).save(
                filepath=filepath,
                overwrite=overwrite,
                include_optimizer=include_optimizer,
                save_format=save_format,
                signatures=signatures,
                options=options,
                save_traces=save_traces,
            )

    def save_weights(
        self,
        filepath: str,
        overwrite: bool = True,
        save_format: str = None,
        options: tf.saved_model.SaveOptions = None,
    ):
        with file_util.save_file(filepath) as path:
            super(BaseModel, self).save_weights(filepath=path,overwrite=overwrite,save_format=save_format,options=options)
        
    def load_weights(
            self,
            filepath,
            by_name=False,
            skip_mismatch=False,
            options=None,
    ):
        with file_util.read_file(filepath) as path:
            super().load_weights(filepath=path, by_name=by_name, skip_mismatch=skip_mismatch, options=options)

    def make(
            self,
            input_shape = [None],
            predictions_shape = [None],
            batch_size = None,
            **kwargs
    ):
        inputs = tf.keras.Input(shape=input_shape, batch_size=batch_size, dtype=tf.float32)
        inputs_length = tf.keras.Input(shape=[], batch_size=batch_size, dtype=tf.int32)
        predictions = tf.keras.Input(shape=predictions_shape, batch_size=batch_size, dtype=tf.int32)
        predictions_length = tf.keras.Input(shape=[], batch_size=batch_size, dtype=tf.int32)

        outputs = self(
            TrainInput(
                inputs=inputs,
                inputs_length=inputs_length,
                predictions=predictions,
                predictions_length=predictions_length,
            ),
            training=False,
        )
        return outputs
    
    def compile(
        self,
        loss,
        optimizer,
        run_eagerly=None,
        **kwargs,
    ):
        optimizer = tf.keras.optimizers.get(optimizer)
        super().compile(optimizer=optimizer, loss=loss, run_eagerly=run_eagerly, **kwargs)

    def call(self, inputs, training=False, mask=None):
        raise NotImplementedError("The call method is not implemented in the base model.")
    
    def _train_step(self, data):
        x = data[0]
        y, _ = data[1]["labels"], data[1]["labels_length"]

        with tf.GradientTape() as tape:
            tape.watch(x["inputs"])
            outputs = self(x, training=True)
            tape.watch(outputs["logits"])
            y_pred = outputs["logits"]
            tape.watch(y_pred)
            loss = self.compute_loss(y, y, y_pred)
            gradients = tape.gradient(loss, self.trainable_variables)

        return gradients
    
    def train_step(self, data):
        gradients = self._train_step(data)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return self.compute_metrics(data)
    
    def _test_step(self, data):
        x = data[0]
        y, _ = data[1]["labels"], data[1]["labels_length"]

        outputs = self(x, training=False)
        y_pred = outputs["logits"]
        return self.compute_loss(y, y, y_pred)
    
    def test_step(self, data):
        self._test_step(data)
        return self.compute_metrics(data)