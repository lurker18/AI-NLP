import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import pandas as pd
import json
import warnings
import nltk
import numpy as np
import evaluate
import torch
from datasets import load_dataset, Dataset, DatasetDict
from peft import LoraConfig
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer, TrainingArguments,)
from tqdm import tqdm
from trl import SFTTrainer

os.environ["WANDB_DISABLED"] = "true"
warnings.filterwarnings("ignore")

base_folder = "/media/lurker18/HardDrive/HuggingFace/models/Microsoft/Phi"
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
    load_in_8bit = True,
    bnb_4bit_quant_type = "nf4",
    bnb_4bit_compute_dtype = "float16",
    bnb_4bit_use_double_quant = False,
)

# 4. Select the Microsoft's Phi-2 model
model = AutoModelForCausalLM.from_pretrained(
    base_folder + "/Phi-2",
    quantization_config = bnb_config,
    device_map = 'auto',
    use_auth_token = False,
)
model.config.pretraining_tp = 1
peft_config = LoraConfig(
    r = 32,
    lora_alpha = 16,
    bias = "none",
    lora_dropout = 0.05,
    task_type = "CAUSAL_LM",
)

# 4.1 Select the tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_folder + "/Phi-2", truncation = True, padding = True)
tokenizer.pad_token = tokenizer.eos_token

# %%
# We prefix our tasks with "answer the question"
prefix = "Please answer this question: "
# Define the preprocessing function
def preprocess_function(examples):
   """Add prefix to the sentences, tokenize the text, and set the labels"""
   # The "inputs" are the tokenized answer:
   inputs = [prefix + doc for doc in examples["question"]]
   model_inputs = tokenizer(inputs, max_length = 128, truncation = True)
  
   # The "labels" are the tokenized outputs:
   labels = tokenizer(text_target = examples["answer"], 
                      max_length = 512,         
                      truncation = True)

   model_inputs["labels"] = labels["input_ids"]
   return model_inputs

# Map the preprocessing function across our dataset
tokenized_dataset = health_dataset_dict.map(preprocess_function, batched = True)

metric = evaluate.load("rouge")

def compute_metrics(eval_preds):
   preds, labels = eval_preds

   # decode preds and labels
   labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
   decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens = True)
   decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens = True)

   # rougeLSum expects newline after each sentence
   decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
   decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

   result = metric.compute(predictions = decoded_preds, 
                           references = decoded_labels, 
                           use_stemmer = True)
  
   return result


training_arguments = TrainingArguments(
    output_dir = "./Results/Phi-2",
    num_train_epochs = 4,
    per_device_train_batch_size = 2,
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

model.config.use_cache = False


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


# %%
import sacrebleu
import numpy as np

# Function to compute BLEU scores
def compute_bleu(preds, labels):
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Prepare references for sacrebleu
    decoded_labels = [[label] for label in decoded_labels]

    # Compute BLEU score
    bleu = sacrebleu.corpus_bleu(decoded_preds, decoded_labels)

    return bleu.score

# Assuming you have the trainer object from the previous code
eval_results = trainer.predict(tokenized_dataset["test"])
preds = eval_results.predictions
labels = eval_results.label_ids

# Replace -100 in labels as tokenizer.pad_token_id
labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

# Compute BLEU score
bleu_score = compute_bleu(preds, labels)
print(f"BLEU score: {bleu_score}")