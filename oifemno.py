from transformers import AutoTokenizer
from dotenv import load_dotenv
load_dotenv()
import os

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", token=os.getenv("HF_TOKEN"))
tokenizer.add_special_tokens({"pad_token": "[PAD]"})


text = "chapter fourteen mount olympus wretched in spirit groaning under the feeling of insult self condemning and ill satisfied in every way bold returned to his london lodgings"
res = tokenizer.encode(text)
print(res)
print(len(res))