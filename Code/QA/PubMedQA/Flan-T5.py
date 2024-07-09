import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from typing import Optional
import nltk
import pandas as pd
import re
import evaluate
import warnings
import numpy as np
import torch
from datasets import load_dataset, Dataset, DatasetDict
from peft import LoraConfig, TaskType, prepare_model_for_kbit_training, get_peft_model
from transformers import (T5ForConditionalGeneration, T5Tokenizer, BitsAndBytesConfig, GenerationConfig, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, Seq2SeqTrainer)
import tqdm
from tqdm import tqdm
import rouge_score
import sacrebleu
import numpy as np
import tensorrt as trt
from Code.utils import convert_format_df

os.environ["WANDB_DISABLED"] = "true"
warnings.filterwarnings("ignore")

base_folder = "/mnt/nvme01/huggingface/models/Google/"

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
model = T5ForConditionalGeneration.from_pretrained(
    base_folder + "flan-t5-large",
    quantization_config = bnb_config,
    torch_dtype = torch.bfloat16,
    device_map = "auto",
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
    task_type = TaskType.SEQ_2_SEQ_LM, # FLAN-T5
    target_modules=['q','v',]
)
model = get_peft_model(model, peft_config)
print(print_number_of_trainable_model_parameters(model))

# 4.1 Select the tokenizer
tokenizer = T5Tokenizer.from_pretrained(base_folder + "flan-t5-large")
data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer, model = model)

prefix = "Please answer this question: "
def preprocess_function(examples):
   """Add prefix to the sentences, tokenize the text, and set the labels"""
   # The "inputs" are the tokenized answer:
   inputs = [prefix + doc + ' ' + str(context).replace("[", '').replace("]", "") for doc, context in zip(examples["question"], examples['context'])]
   model_inputs = tokenizer(inputs, max_length = 512, truncation = True)
  
   # The "labels" are the tokenized outputs:
   labels = tokenizer(text_target = examples["answer"], 
                      max_length = 2048,         
                      truncation = True)

   model_inputs["labels"] = labels["input_ids"]
   return model_inputs

tokenized_dataset = health_dataset_dict.map(preprocess_function, batched = True)

metric = evaluate.load("rouge")

def compute_metrics(eval_preds):
   preds, labels = eval_preds

   # decode preds and labels
   labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
   decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
   decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

   # rougeLSum expects newline after each sentence
   decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
   decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

   result = metric.compute(predictions = decoded_preds, references = decoded_labels, use_stemmer = True)
  
   return result

BATCH_SIZE = 32
PER_DEVICE_EVAL_BATCH = 32

training_arguments = Seq2SeqTrainingArguments(
    output_dir = "./Results/PubMedQA/Flan-T5",
    num_train_epochs = 5,
    gradient_accumulation_steps = 1,
    optim = "paged_adamw_32bit",
    lr_scheduler_type = "cosine",
    save_strategy = "epoch",
    evaluation_strategy = "epoch",
    logging_steps = 500,
    logging_strategy = "steps",
    learning_rate = 2e-4,
    warmup_ratio = 0.03,  
    disable_tqdm = False,
    report_to = None,
    per_device_train_batch_size = BATCH_SIZE,
    per_device_eval_batch_size = PER_DEVICE_EVAL_BATCH,
    predict_with_generate = True,
)

# 5. Training the model
trainer = Seq2SeqTrainer(
   model = model,
   args = training_arguments,
   train_dataset = tokenized_dataset["train"],
   eval_dataset = tokenized_dataset["test"],
   tokenizer = tokenizer,
   data_collator = data_collator,
   compute_metrics = compute_metrics
)

trainer.train()

# Function to compute BLEU scores
def compute_bleu(preds, labels):
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens = True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens = True)

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

