import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import pandas as pd
import json
import warnings

import torch
from datasets import load_dataset, Dataset, DatasetDict
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer, TrainingArguments,)
from tqdm import tqdm
import tensorrt as trt
from trl import SFTTrainer

os.environ["WANDB_DISABLED"] = "true"
warnings.filterwarnings("ignore")

base_folder = "/media/lurker18/HardDrive/HuggingFace/models/Google/"

# 1. Load the Dataset
raw_data = load_dataset("lavita/MedQuAD", split = 'train')
raw_data.set_format(type = 'pandas')
raw_df = raw_data[:]
temp = raw_df[~raw_df['answer'].isnull() & (raw_df['answer'] != '')]
raw_data = Dataset.from_pandas(temp)
temp_dataset = raw_data.train_test_split(test_size = 0.2)
dataset = temp_dataset['train'].train_test_split(test_size = 0.125)
dataset.set_format(type = 'pandas')
temp_dataset.set_format(type = 'pandas')
train_data = dataset['train'][:]
val_data = dataset['test'][:]
test_data = temp_dataset['test'][:]
df_train = train_data[['question', 'answer']]
df_val = val_data[['question', 'answer']]
df_test = test_data[['question', 'answer']]

train_dataset = Dataset.from_pandas(df_train)
val_dataset = Dataset.from_pandas(df_val)
test_dataset = Dataset.from_pandas(df_test)

health_dataset_dict = DatasetDict({
    'train': train_dataset,
    'validation': val_dataset,
    'test': test_dataset
})
# %%

def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    return f"trainable model parameters: {trainable_model_params}\nall model parameters: {all_model_params}\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%"


# 3. Set the quantization settings
bnb_config = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_quant_type = "nf4",
    bnb_4bit_compute_dtype = torch.bfloat16,
    bnb_4bit_use_double_quant = False,
)

# 4. Select the Microsoft's Phi-2 model
model = AutoModelForCausalLM.from_pretrained(
    base_folder + "gemma-7b-Instruct",
    quantization_config = bnb_config,
    attn_implementation = "flash_attention_2",
    device_map = "auto",
    use_auth_token = False,
)
model.config.use_cache = False # silence the warnings. Please re-enable for inference!
model.config.pretraining_tp = 1
model = prepare_model_for_kbit_training(model)

peft_config = LoraConfig(
    lora_alpha = 16,
    lora_dropout = 0.1,
    r = 64,
    bias = "none",
    task_type = "CAUSAL_LM",
    target_modules = ['o_proj', 'q_proj', 'up_proj', 'v_proj', 'k_proj', 'down_proj', 'gate_proj']
)
model = get_peft_model(model, peft_config)

# 4.1 Select the tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_folder + "gemma-7b-Instruct", truncation = True, padding = "max_length")
tokenizer.padding_side = 'right'
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_eos_token = True
tokenizer.add_bos_token, tokenizer.add_eos_token

training_arguments = TrainingArguments(
    output_dir = "./Results/Gemma-7B-Instruct",
    num_train_epochs = 4,
    per_device_train_batch_size = 4,
    gradient_accumulation_steps = 1,
    optim = "paged_adamw_32bit",
    save_strategy = "epoch",
    logging_steps = 100,
    logging_strategy = "steps",
    learning_rate = 2e-4,
    fp16 = False,
    bf16 = False,
    group_by_length = True,
    disable_tqdm = False,
    report_to = None
)

# 5. Training the model
trainer = SFTTrainer(
    model = model,
    train_dataset = dataset,
    peft_config = peft_config,
    dataset_text_field = "text",
    max_seq_length = 2048,
    tokenizer = tokenizer,
    args = training_arguments,
    packing = False,
)

trainer.train()

# 6. Test and compare the non-fine-tuned model against the fine-tuned Phi-2 model
print(medquad.iloc[2050, :]['text'])
print(medquad.iloc[2050, :]['label'])

# Fine-tuned Gemma-7b-Instruct model performance
inputs = tokenizer('''Question: What is (are) Trigeminal Neuralgia ?\n Output:''', return_tensors = 'pt', return_attention_mask = False)
outputs = model.generate(**inputs, max_length = 200)
text = tokenizer.batch_decode(outputs[0], skip_special_tokens = True)
print(''.join(text))

# Non-Fine-tuned Gemma-7b-Instruct model performance
torch.set_default_device("cuda")
model_test = AutoModelForCausalLM.from_pretrained(base_folder + "gemma-7b-Instruct", torch_dtype = "auto")
tokenizer = AutoTokenizer.from_pretrained(base_folder + "gemma-7b-Instruct", truncation = True, padding = True)
inputs = tokenizer('''Question: What is (are) Trigeminal Neuralgia ?\n Output:''', return_tensors = 'pt', return_attention_mask = False)
outputs = model_test.generate(**inputs, max_length = 100)
text = tokenizer.batch_decode(outputs)[0]
print(text)