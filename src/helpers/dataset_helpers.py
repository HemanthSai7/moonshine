from src.speech_featurizer import SpeechFeaturizer
from omegaconf import DictConfig
from dotenv import load_dotenv

import os
import tensorflow as tf
from transformers import AutoTokenizer
load_dotenv()


logger = tf.get_logger()


def prepare_featurizers(
    config: DictConfig,
):
    speech_config = config.speech_config
    feature_extractor = SpeechFeaturizer(**dict(speech_config))

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", token=os.getenv("HF_TOKEN"))
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    
    return feature_extractor, tokenizer
