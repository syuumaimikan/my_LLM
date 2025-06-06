import os
import torch
import pandas as pd
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig
from trl import SFTTrainer

# データの読み込み
data = pd.read_csv("conversation_data.csv")  # データファイルのパスを適切に設定してください

# モデルとトークナイザーの設定
model_name = "elyza/ELYZA-japanese-Llama-2-7b-instruct"
base_dir = "./saved_models"
model_path = "ELYZA_chat_model"
tokenizer_path = "tokenizer"

# プロンプト設定
NAME = "****"  # あなたの名前
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT = "あなたは{name}という名前の日本人です。今は{time}です。以下の会話に{name}として答えてください"

# データセットの作成
dataset = pd.Series(["" for _ in range(len(data))])

# GPUの設定
device = "cuda" if torch.cuda.is_available() else "cpu"
if torch.backends.mps.is_available():
    device = "mps"

# データセットの作成
for i in range(len(data)):
    if data.loc[i, "system_prompt"] == "":
        system = B_SYS + DEFAULT_SYSTEM_PROMPT.format(name=NAME, time=data.loc[i, "timestamp"]) + E_SYS
    else:
        system = data.loc[i, "system_prompt"]
    
    prompt = "{bos_token}{b_inst} {system}{prompt} {e_inst} {output} {eos_token}".format(
        bos_token=tokenizer.bos_token,
        b_inst=B_INST,
        system=system,
        prompt=data.loc[i, "input_1"],
        e_inst=E_INST,
        output=data.loc[i, "output_1"],
        eos_token=tokenizer.eos_token
    )
    
    for j in range(data.loc[i, "input_num"]-1):
        prompt += "{bos_token}{b_inst} {prompt} {e_inst} {output} {eos_token}".format(
            bos_token=tokenizer.bos_token,
            b_inst=B_INST,
            prompt=data.loc[i, f"input_{j+2}"],
            e_inst=E_INST,
            output=data.loc[i, f"output_{j+2}"],
            eos_token=tokenizer.eos_token
        )
    
    dataset[i] = prompt

# モデルの設定
compute_dtype = getattr(torch, "float16")
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)

# モデルとトークナイザーの読み込み
model = AutoModelForCausalLM.from_pretrained(
    os.path.join(base_dir, model_path),
    quantization_config=quant_config,
    device_map={"": 0}
)
model.config.use_cache = False
model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained(
    os.path.join(base_dir, tokenizer_path),
    trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# LoRA設定
peft_params = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)

# トレーニング設定
training_params = TrainingArguments(
    output_dir=os.path.join(base_dir, "train_output"),
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    save_steps=25,
    logging_steps=25,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to="tensorboard"
)

# トレーナーの設定と学習
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_params,
    dataset_text_field="text",
    max_seq_length=None,
    tokenizer=tokenizer,
    args=training_params,
    packing=False,
)

trainer.train()

# モデルの保存
trainer.model.save_pretrained(os.path.join(base_dir, "chat_model"))
trainer.tokenizer.save_pretrained(os.path.join(base_dir, "tokenizer"))
