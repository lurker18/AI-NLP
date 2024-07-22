import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3,4"
import re
import pandas as pd
import warnings
import torch
from datasets import load_dataset, Dataset, DatasetDict
from peft import LoraConfig, AutoPeftModelForCausalLM, prepare_model_for_kbit_training, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer, TrainingArguments, GenerationConfig)
import tqdm
from tqdm import tqdm
import tensorrt as trt
from trl import SFTTrainer
from Code.utils import convert_format_df

os.environ["WANDB_DISABLED"] = "true"
warnings.filterwarnings("ignore")

base_folder = "/mnt/nvme01/huggingface/models/Facebook/"

# 1. Load the Dataset
dataset = load_dataset("bigbio/pubmed_qa")
dataset.set_format(type = 'pandas')
train_data = dataset['train'][:]
val_data = dataset['validation'][:]

data = pd.concat([train_data, val_data]).reset_index()
del data['index']
raw_dataset = Dataset.from_pandas(data)
temp_dataset = raw_dataset.train_test_split(test_size = 0.2)
dataset = temp_dataset['train'].train_test_split(test_size = 0.125)
dataset.set_format(type = 'pandas')
temp_dataset.set_format(type = 'pandas')
train_data = dataset['train'][:]
test_data = temp_dataset['test'][:]

# 2. Preset the the Instruction-based prompt template
train_df, train_hf = convert_format_df(train_data, data_name = 'pubmedqa')
val_df, val_hf = convert_format_df(val_data, data_name = 'pubmedqa')
test_df, test_hf = convert_format_df(val_data, data_name = 'pubmedqa')

health_dataset_dict = DatasetDict({
    'train': train_hf,
    'validation': val_hf,
    'test': test_hf
})


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

# 4. Select the MistralAI's Mistral-7B-Instruct model
model = AutoModelForCausalLM.from_pretrained(
    base_folder + "opt-2.7b",
    quantization_config = bnb_config,
    attn_implementation = "flash_attention_2",
    torch_dtype = torch.bfloat16,
    device_map = "balanced",
    use_auth_token = True,
    use_safetensors = True,
)
model.config.use_cache = False
model.config.pretraining_tp = 1
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)
peft_config = LoraConfig(
    lora_alpha = 32,
    lora_dropout = 0.05,
    r = 16,
    bias = "none",
    task_type = "CAUSAL_LM",
    target_modules = ["q_proj", "v_proj"]
)
model = get_peft_model(model, peft_config)
print(print_number_of_trainable_model_parameters(model))

# 4.1 Select the tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_folder + "opt-2.7b", padding = "max_length", truncation = True, max_length = 2048)

training_arguments = TrainingArguments(
    output_dir = "./Results/PubMedQA/opt-2.7b",
    num_train_epochs = 4,
    per_device_train_batch_size = 24,
    gradient_accumulation_steps = 1,
    optim = "paged_adamw_32bit",
    lr_scheduler_type = "cosine",
    save_strategy = "epoch",
    logging_steps = 100,
    logging_strategy = "steps",
    learning_rate = 2e-4,
    bf16 = True,
    fp16 = False, 
    group_by_length = True,
    disable_tqdm = False,
    report_to = None
)

# 5. Training the model
trainer = SFTTrainer(
    model = model,
    train_dataset = train_hf,
    eval_dataset = val_hf,
    peft_config = peft_config,
    dataset_text_field = "text",
    max_seq_length = 2048,
    tokenizer = tokenizer,
    args = training_arguments,
    packing = False,
)

trainer.train()

# Function to compute BLEU scores
# def compute_bleu(preds, labels):
#     decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens = True)
#     decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens = True)

#     # Prepare references for sacrebleu
#     decoded_labels = [[label] for label in decoded_labels]

#     # Compute BLEU score
#     bleu = sacrebleu.corpus_bleu(decoded_preds, decoded_labels)

#     return bleu.score

# # Assuming you have the trainer object from the previous code
# eval_results = trainer.predict(tokenized_dataset["test"])
# preds = eval_results.predictions
# labels = eval_results.label_ids

# # Replace -100 in labels as tokenizer.pad_token_id
# labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

# # Compute BLEU score
# bleu_score = compute_bleu(preds, labels)
# print(f"BLEU score: {bleu_score}")

