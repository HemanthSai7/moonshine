from src.utils import data_util
from src.dataset import get
from src.helpers import prepare_featurizers

from dotenv import load_dotenv
from omegaconf import DictConfig

import os
import hydra
import tensorflow as tf

load_dotenv()

# f = SpeechFeaturizer()

# file = tf.io.read_file("/home/hemanth/InHouseODV/odvModelTraining.hemanth.saigarladinne/data/sensory/SensoryDataset/nz-helloLloyd+ac@all-raw-1-INF-AMT_en_IN_20220706_HS.A11IXNHDZWA9FG.1659055016765.wav")
# audio = read_raw_audio(file, sample_rate=16000)
# audio = f([audio, tf.shape(audio)[0]], training=False)
# print(audio)

@hydra.main(config_path="config", config_name="config")
def main(
    config: DictConfig,
    batch_size: int = 2,
    mxp: int = "none",
    spx: int = None,
    jit_compile: bool = False,
):
    tf.keras.backend.clear_session()
    # env_util.setup_seed()
    # env_util.setup_mxp(mxp=mxp)

    speech_featurizer, tokenizer = prepare_featurizers(config)

    train_dataset = get(
        speech_featurizer=speech_featurizer,
        tokenizer=tokenizer,
        dataset_config=config.learning_config.train_dataset_config,
    )

    valid_dataset = get(
        speech_featurizer=speech_featurizer,
        tokenizer=tokenizer,
        dataset_config=config.learning_config.eval_dataset_config,
    )
    
    train_data_loader = train_dataset.create(batch_size=batch_size)
    valid_data_loader = valid_dataset.create(batch_size=batch_size)

    for data in valid_data_loader:
        print(data)
        break

if __name__ == "__main__":
    main()
    