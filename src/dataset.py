from src.schemas import TrainInput, TrainLabel
from src import DatasetConfig
from src.speech_featurizer import SpeechFeaturizer
from src.utils import (
    data_util,
    file_util,
    math_util,
)

from transformers import AutoTokenizer

import os
import json
import tqdm
import numpy as np
import tensorflow as tf

logger = tf.get_logger()

def get(
    tokenizer: AutoTokenizer,
    speech_featurizer: SpeechFeaturizer,
    dataset_config: DatasetConfig,
):
    return ASRSliceDataset(
        speech_featurizer=speech_featurizer,
        tokenizer=tokenizer,
        stage=dataset_config.stage,
        data_paths=list(dataset_config.data_paths),
    )

def get_shape(
    config,
    *datasets,
    batch_size: int = None,
):
    batch_size = batch_size or config.learning_config.running_config.batch_size

    max_input_length, max_label_length = None, None

    input_shape = [max_input_length]
    predictions_shape = [max_label_length]
    label_shape = [max_label_length]

    padded_shapes = (
        TrainInput(
            inputs=tf.TensorShape(input_shape),
            inputs_length=tf.TensorShape([]),
            predictions=tf.TensorShape(predictions_shape),
            predictions_length=tf.TensorShape([]),
        ),
        TrainLabel(
            labels=tf.TensorShape(label_shape),
            labels_length=tf.TensorShape([]),
        )
    )

    return dict(
        batch_size=batch_size,
        input_shape=input_shape,
        predictions_shape=predictions_shape,
        label_shape=label_shape,
        padded_shapes=padded_shapes,
    )

BUFFER_SIZE = 100
AUTOTUNE = int(os.environ.get("AUTOTUNE", tf.data.AUTOTUNE))

class BaseDataset:
    def __init__(
        self,
        data_paths: list,
        cache: bool = False,
        shuffle: bool = False,
        buffer_size: int = BUFFER_SIZE,
        indefinite: bool = False,
        drop_remainder: bool = True,
        enabled: bool = True,
        metadata: str = None,
        sample_rate: int = 16000,
        stage: str = "train",
        name: str = "base_dataset",
        **kwargs,
    ):
        self.data_paths = data_paths or []
        if not isinstance(self.data_paths, list):
            raise ValueError("data_paths must be a list of string paths")
        self.cache = cache
        self.shuffle = shuffle
        self.buffer_size = buffer_size
        self.stage = stage
        self.enabled = enabled
        self.drop_remainder = drop_remainder
        self.indefinite = indefinite
        self.total_steps = None
        self.metadata = metadata
        self.sample_rate = sample_rate
        self.name = name

    def parse(self, *args, **kwargs):
        raise NotImplementedError()
    
    def create(self, *args, **kwargs):
        raise NotImplementedError()
    
class ASRDataset(BaseDataset):
    def __init__(
        self,
        stage: str,
        tokenizer: AutoTokenizer,
        speech_featurizer: SpeechFeaturizer,
        data_paths: list,
        cache: bool = False,
        shuffle: bool = False,
        indefinite: bool = True,
        drop_remainder: bool = True,
        enabled: bool = True,
        metadata: str = None,
        buffer_size: int = BUFFER_SIZE,
        sample_rate: int = 16000,
        training=False,
        name: str = "asr_dataset",
        **kwargs,
    ):
        super(ASRDataset, self).__init__(
            data_paths=data_paths,
            cache=cache,
            shuffle=shuffle,
            buffer_size=buffer_size,
            indefinite=indefinite,
            drop_remainder=drop_remainder,
            enabled=enabled,
            metadata=metadata,
            sample_rate=sample_rate,
            stage=stage,
            name=name,
        )
        self.entries = []
        self.tokenizer = tokenizer
        self.speech_featurizer = speech_featurizer
        self.training = training

    def read_entries(self):
        if hasattr(self, "entries") and len(self.entries) > 0:
            return
        self.data_paths = file_util.preprocess_paths(self.data_paths, enabled=self.enabled, check_exists=True)
        for file_path in self.data_paths:
            logger.info(f"Reading entries from {file_path}")
            with tf.io.gfile.GFile(file_path, "r") as f:
                for line in f.read().splitlines()[1:]:
                    self.entries.append(line.split("\t", 2))
        self.entries = np.array(self.entries)
        if self.shuffle:
            np.random.shuffle(self.entries)
        self.total_steps = len(self.entries)
        self.num_entries = self.total_steps
        logger.info(f"Total entries: {self.num_entries}")

    def _process_item(self, path: tf.Tensor, audio: tf.Tensor, transcript: tf.Tensor):
        inputs = data_util.read_raw_audio(audio, sample_rate=self.sample_rate)
        inputs_length = tf.cast(tf.shape(inputs)[0], tf.int32)
        inputs, inputs_length = self.speech_featurizer((inputs, inputs_length), training=self.training)

        transcript_str = tf.strings.as_string(transcript)
        transcript_str = tf.ensure_shape(transcript_str, [])

        def tokenize_transcript(text):
            text_str = text.numpy().decode("utf-8")
            return np.array(self.tokenizer.encode(text_str, add_special_tokens=True), dtype=np.int32)
        
        labels = tf.py_function(
            func=tokenize_transcript,
            inp=[transcript_str],
            Tout=tf.int32,
        )

        predictions = labels
        predictions_length = tf.shape(predictions)[0]

        labels_length = tf.cast(tf.shape(labels)[0], tf.int32)

        return path, inputs, inputs_length, predictions, predictions_length, labels, labels_length
    
    def parse(self, path: tf.Tensor, audio: tf.Tensor, transcript: tf.Tensor):
        (
            _,
            inputs,
            inputs_length,
            predictions,
            predictions_length,
            labels,
            labels_length,
        ) = self._process_item(path=path, audio=audio, transcript=transcript)

        return (
            TrainInput(inputs=inputs, inputs_length=inputs_length, predictions=predictions, predictions_length=predictions_length),
            TrainLabel(labels=labels, labels_length=labels_length),
        )
    
    def process(self, dataset: tf.data.Dataset, batch_size: int, padded_shapes=None):
        if self.cache:
            dataset = dataset.cache()

        dataset = dataset.map(self.parse, num_parallel_calls=AUTOTUNE, deterministic=False)
        self.total_steps = math_util.get_num_batches(self.num_entries, batch_size, drop_remainders=self.drop_remainder)

        if self.shuffle:
            dataset = dataset.shuffle(buffer_size=self.buffer_size, reshuffle_each_iteration=True)

        if self.indefinite and self.total_steps:
            dataset = dataset.repeat(self.total_steps)

        if padded_shapes is None:
            padded_shapes = (
                TrainInput(
                    inputs=tf.TensorShape([None]),
                    inputs_length=tf.TensorShape([]),
                    predictions=tf.TensorShape([None]),
                    predictions_length=tf.TensorShape([]),
                ),
                TrainLabel(
                    labels=tf.TensorShape([None]),
                    labels_length=tf.TensorShape([]),
                )
            )

        dataset = dataset.padded_batch(
            batch_size=batch_size,
            padded_shapes=padded_shapes,
            padding_values = (
                TrainInput(
                    inputs=32000.0,
                    inputs_length=tf.constant(32000, dtype=tf.int32),
                    predictions=tf.constant(32000, dtype=tf.int32),
                    predictions_length=tf.constant(32000, dtype=tf.int32),
                ),
                TrainLabel(
                    labels=tf.constant(32000, dtype=tf.int32),
                    labels_length=tf.constant(32000, dtype=tf.int32),
                ),
            ),
            drop_remainder=self.drop_remainder,
        )

        dataset = dataset.prefetch(buffer_size=AUTOTUNE)
        return dataset

class ASRSliceDataset(ASRDataset):
    def load(self, record):
        audio = tf.py_function(
            lambda path: data_util.load_and_convert_to_wav(path.numpy().decode("utf-8")).numpy(),
            inp = [record[0]],
            Tout=tf.string,
        )

        return record[0], audio, record[2]
    
    def create(self, batch_size: int, padded_shapes=None):
        if not self.enabled:
            return None
        self.read_entries()
        if not self.total_steps or self.total_steps == 0:
            return None
        
        dataset = tf.data.Dataset.from_tensor_slices(self.entries)
        options = tf.data.Options()
        options.deterministic = False
        dataset = dataset.with_options(options)
        dataset = dataset.map(self.load, num_parallel_calls=AUTOTUNE, deterministic=False)
        return self.process(dataset, batch_size=batch_size, padded_shapes=padded_shapes)