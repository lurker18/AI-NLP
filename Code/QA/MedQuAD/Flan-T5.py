# %%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import re
import nltk
import evaluate
import pandas as pd
import numpy as np
import warnings
import tqdm
import torch
from datasets import load_dataset, Dataset, DatasetDict
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from transformers import T5Tokenizer, DataCollatorForSeq2Seq, BitsAndBytesConfig
from transformers import T5ForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer

os.environ["WANDB_DISABLED"] = "true"
warnings.filterwarnings("ignore")

base_folder = "D:/HuggingFace/models/Google/"

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

bnb_config = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_quant_type = "nf4",
    bnb_4bit_compute_dtype = torch.bfloat16,
    bnb_4bit_use_double_quant = False,
)

lora_config = LoraConfig(
    r = 16, # Rank
    lora_alpha = 32,
    target_modules = ["q", "v"],
    lora_dropout = 0.05,
    bias = "none",
    task_type = TaskType.SEQ_2_SEQ_LM # FLAN-T5
)

tokenizer = T5Tokenizer.from_pretrained(base_folder + "flan-t5-large")
model = T5ForConditionalGeneration.from_pretrained( base_folder + "flan-t5-large",
                                                    quantization_config = bnb_config,
                                                    torch_dtype = torch.bfloat16,
                                                    device_map = "auto",
                                                    use_auth_token = True,
                                                    use_safetensors = True,)
model.config.use_cache = False
model.config.pretraining_tp = 1
model.gradient_checkpointing_enable()
data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer,
                                       model = model)
model = prepare_model_for_kbit_training(model)
peft_model = get_peft_model(model, lora_config)
print(print_number_of_trainable_model_parameters(peft_model))

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

L_RATE = 1e-4
BATCH_SIZE = 32
PER_DEVICE_EVAL_BATCH = 32
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 5

# Set up training arguments
training_args = Seq2SeqTrainingArguments(
   output_dir = "./Results/MedQuAD/Flan-T5",
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

trainer = Seq2SeqTrainer(
   model = peft_model,
   args = training_args,
   train_dataset = tokenized_dataset["train"],
   eval_dataset = tokenized_dataset["test"],
   tokenizer = tokenizer,
   data_collator = data_collator,
   compute_metrics = compute_metrics
)
# %%
# Free GPU memory
torch.cuda.empty_cache()
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

# %%
