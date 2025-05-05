from transformers import AutoTokenizer
from dotenv import load_dotenv
# from src.model.decoder import TextEmbeddings
load_dotenv()
import os

import tensorflow as tf

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", token=os.getenv("HF_TOKEN"))
tokenizer.add_special_tokens({"pad_token": "[PAD]"})


text1 = "chapter fourteen mount olympus wretched in spirit groaning under the feeling of insult self condemning and ill satisfied in every way bold returned to his london lodgings"
text2 = "hi there"
text = (text1, text2)
# res = tokenizer.batch_encode_plus(
#     text,
#     add_special_tokens=True,
#     max_length=512,
#     padding="max_length",
#     truncation=True,
#     return_tensors="tf",
# )
# print(res)
# print(len(res))
# print(tokenizer.vocab_size)
print(list(tokenizer.get_vocab().keys())[list(tokenizer.get_vocab().values()).index(32000)])  # Prints george
# tgt = TextEmbeddings(
#     vocab_size=tokenizer.vocab_size,
#     d_model=288,
#     name="text_embeddings",
# )

# emb, query, key = tgt(tf.expand_dims(tf.constant(res), axis=0))
# print(emb)
# print(query)
# print(key)