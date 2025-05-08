from src.utils import data_util
from src.dataset import get, get_shape
from src.helpers import prepare_featurizers
from src.model import Moonshine

from dotenv import load_dotenv
from omegaconf import DictConfig

import os
import hydra
import tensorflow as tf

load_dotenv()
logger = tf.get_logger()

# f = SpeechFeaturizer()

# file = tf.io.read_file("/home/hemanth/InHouseODV/odvModelTraining.hemanth.saigarladinne/data/sensory/SensoryDataset/nz-helloLloyd+ac@all-raw-1-INF-AMT_en_IN_20220706_HS.A11IXNHDZWA9FG.1659055016765.wav")
# audio = read_raw_audio(file, sample_rate=16000)
# audio = f([audio, tf.shape(audio)[0]], training=False)
# print(audio)

@hydra.main(config_path="config", config_name="config")
def main(
    config: DictConfig,
    batch_size: int = 2,  # Reduced default batch size
    mxp: int = "none",
    spx: int = None,
    jit_compile: bool = False,
):
    # physical_devices = tf.config.list_physical_devices('GPU')
    # if len(physical_devices) > 0:
    #     for device in physical_devices:
    #         try:
    #             tf.config.experimental.set_memory_growth(device, True)
    #             print(f"Memory growth enabled for {device}")
    #         except:
    #             print(f"Could not set memory growth for {device}")
            
    # if len(physical_devices) > 0:
    #     try:
    #         tf.config.set_logical_device_configuration(
    #             physical_devices[0],
    #             [tf.config.LogicalDeviceConfiguration(memory_limit=4096)]  # Further reduced to 4GB
    #         )
    #         print("GPU memory limited to 4GB")
    #     except Exception as e:
    #         print(f"Could not limit GPU memory: {e}")
    
    # os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    # os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
    
    # tf.config.optimizer.set_jit(False)
    
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

    shapes = get_shape(
        config,
        train_dataset,
        valid_dataset,
        batch_size=batch_size or config.learning_config.running_config.batch_size,
    )
    
    train_data_loader = train_dataset.create(batch_size=batch_size)
    valid_data_loader = valid_dataset.create(batch_size=batch_size)

    for data in valid_data_loader:
        print(data)
        break
    moonshine = Moonshine(**config.model_config, vocab_size=tokenizer.vocab_size)
    moonshine.make(**shapes)
    
    dummy_inputs = tf.zeros((batch_size, 100), dtype=tf.float32)
    dummy_inputs_length = tf.zeros((batch_size,), dtype=tf.int32)
    dummy_predictions = tf.zeros((batch_size, 30), dtype=tf.int32)
    dummy_predictions_length = tf.zeros((batch_size,), dtype=tf.int32)

    try:
        logger.info("Starting model call...")
        outputs = moonshine({"inputs": dummy_inputs, "inputs_length": dummy_inputs_length, "predictions":dummy_predictions, "predictions_length":dummy_predictions_length}, training=True)
        logger.info("Model call successful")
        logger.info(f"Output shape: {outputs.shape}")
        moonshine.summary(expand_nested=True)
    except Exception as e:
        logger.error(f"Error during model call: {e}")
        raise

if __name__ == "__main__":
    main()