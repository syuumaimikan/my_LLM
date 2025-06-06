import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer

model_name = "elyza/ELYZA-japanese-Llama-2-7b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto",cache_dir ="./saved_models")

tokenizer.save_pretrained("tokenizer")
model.save_pretrained("ELYZA_chat_model")
