from src.utils import file_util

from typing import Union

import json
import tensorflow as tf

logger = tf.get_logger()

__all__ = [
    "DecoderConfig",
    "DatasetConfig",
    "DataConfig",
    "LearningConfig",
    "Config",
]


class DecoderConfig:
    def __init__(self, config: dict = None):
        if not config:
            config = {}
        self.type: str = config.get("type", "wordpiece")

        self.blank_index: int = config.get("blank_index", 0)
        self.pad_token: str = config.get("pad_token", "<pad>")
        self.pad_index: int = config.get("pad_index", -1)
        self.unknown_token: str = config.get("unknown_token", "<unk>")
        self.unknown_index: int = config.get("unknown_index", 0)
        self.bos_token: str = config.get("bos_token", "<s>")
        self.bos_index: int = config.get("bos_index", -1)
        self.eos_token: str = config.get("eos_token", "</s>")
        self.eos_index: int = config.get("eos_index", -1)

        self.beam_width: int = config.get("beam_width", 0)
        self.norm_score: bool = config.get("norm_score", True)
        self.lm_config: dict = config.get("lm_config", {})

        self.model_type: str = config.get("model_type", "unigram")
        self.vocabulary: str = file_util.preprocess_paths(config.get("vocabulary", None))
        self.vocab_size: int = config.get("vocab_size", 1000)
        self.max_token_length: int = config.get("max_token_length", 50)
        self.max_unique_chars: int = config.get("max_unique_chars", None)
        self.num_iterations: int = config.get("num_iterations", 4)
        self.reserved_tokens: list = config.get("reserved_tokens", None)
        self.normalization_form: str = config.get("normalization_form", "NFKC")
        self.keep_whitespace: bool = config.get("keep_whitespace", False)
        self.max_sentence_length: int = config.get("max_sentence_length", 1048576)  # bytes
        self.max_sentencepiece_length: int = config.get("max_sentencepiece_length", 16)  # bytes
        self.character_coverage: float = config.get("character_coverage", 1.0)  # 0.9995 for languages with rich character, else 1.0

        self.train_files = config.get("train_files", [])
        self.eval_files = config.get("eval_files", [])

        for k, v in config.items():
            setattr(self, k, v)


class DatasetConfig:
    def __init__(self, config: dict = None):
        if not config:
            config = {}
        self.name: str = config.get("name", "")
        self.enabled: bool = config.get("enabled", True)
        self.stage: str = config.get("stage", None)
        self.data_paths = config.get("data_paths", None)
        self.shuffle: bool = config.get("shuffle", False)
        self.cache: bool = config.get("cache", False)
        self.drop_remainder: bool = config.get("drop_remainder", True)
        self.buffer_size: int = config.get("buffer_size", 1000)
        self.metadata: str = config.get("metadata", None)
        self.sample_rate: int = config.get("sample_rate", 16000)
        for k, v in config.items():
            setattr(self, k, v)


class DataConfig:
    def __init__(self, config: dict = None):
        if not config:
            config = {}
        self.train_dataset_config = DatasetConfig(config.get("train_dataset_config", {}))
        self.eval_dataset_config = DatasetConfig(config.get("eval_dataset_config", {}))
        self.test_dataset_configs = [DatasetConfig(conf) for conf in config.get("test_dataset_configs", [])]
        _test_dataset_config = config.get("test_dataset_config", None)
        if _test_dataset_config:
            self.test_dataset_configs.append(_test_dataset_config)


class LearningConfig:
    def __init__(self, config: dict = None):
        if not config:
            config = {}
        self.pretrained = file_util.preprocess_paths(config.get("pretrained", None))
        self.optimizer_config: dict = config.get("optimizer_config", {})
        self.gradn_config = config.get("gradn_config", None)
        self.batch_size: int = config.get("batch_size", 2)
        self.ga_steps: int = config.get("ga_steps", None)
        self.num_epochs: int = config.get("num_epochs", 300)
        self.callbacks: list = config.get("callbacks", [])
        for k, v in config.items():
            setattr(self, k, v)


class Config:
    """User config class for training, testing or infering"""

    def __init__(self, data: Union[str, dict], training=True, **kwargs):
        config = data if isinstance(data, dict) else file_util.load_yaml(file_util.preprocess_paths(data), **kwargs)
        self.decoder_config = DecoderConfig(config.get("decoder_config", {}))
        self.model_config: dict = config.get("model_config", {})
        self.data_config = DataConfig(config.get("data_config", {}))
        self.learning_config = LearningConfig(config.get("learning_config", {})) if training else None
        for k, v in config.items():
            setattr(self, k, v)
        logger.info(str(self))

    def __str__(self) -> str:
        def default(x):
            try:
                return {k: v for k, v in vars(x).items() if not str(k).startswith("_")}
            except:  # pylint: disable=bare-except
                return str(x)

        return json.dumps(vars(self), indent=2, default=default)
