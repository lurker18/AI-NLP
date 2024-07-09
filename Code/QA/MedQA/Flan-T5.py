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
from transformers import (T5ForConditionalGeneration, T5Tokenizer, BitsAndBytesConfig, AutoTokenizer, TrainingArguments, GenerationConfig, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, Seq2SeqTrainer)
import tqdm
from tqdm import tqdm
import rouge_score
import tensorrt as trt

os.environ["WANDB_DISABLED"] = "true"
warnings.filterwarnings("ignore")

base_folder = "/mnt/nvme01/huggingface/models/Google/"

# 1. Load the Dataset
dataset = load_dataset("bigbio/med_qa")
train_data = dataset['train']
val_data = dataset['validation']
test_data = dataset['test']

# 2. Preset the the Instruction-based prompt template
def generate_prompt(x):
    answer_idx = 'Nothing'
    if x['answer_idx'] == 'A':
        answer_idx = x['opa']
    elif x['answer_idx'] == 'B':
        answer_idx = x['opb']
    elif x['answer_idx'] == 'C':
        answer_idx = x['opc']
    elif x['answer_idx'] == 'D':
        answer_idx = x['opd']
    elif x['answer_idx'] == 'E':
        answer_idx = x['ope']
    question = '{}\nOptions:\n1. {}\n2. {}\n3. {}\n4. {}\n5. {}\n'.format(x['question'], 
                                                                          x['opa'],
                                                                          x['opb'], 
                                                                          x['opc'], 
                                                                          x['opd'], 
                                                                          x['ope'])
    answer = answer_idx
    prompt = f"Question: {question}\nAnswer: {answer}"
    return prompt

def generate_and_tokenize_prompt(prompt):
    return tokenizer(generate_prompt(prompt), truncation = True)

def convert_format_df(data):
    data_extracted = [
    {'question' : variable['question'], 
     'answer_idx' : variable['answer_idx'], 
     'answer' : variable['answer'],
     'opa' : variable['options'][0]['value'], 
     'opb' : variable['options'][1]['value'], 
     'opc' : variable['options'][2]['value'], 
     'opd' : variable['options'][3]['value'], 
     'ope' : variable['options'][4]['value']}
    for variable in data
    ]
    df = pd.DataFrame(data_extracted)
    df['text'] = df.apply(lambda x: generate_prompt(x), axis = 1)
    data_hf = Dataset.from_pandas(df)
    return df, data_hf

train_df, train_hf = convert_format_df(train_data)
val_df, val_hf = convert_format_df(val_data)
test_df, test_hf = convert_format_df(test_data)

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
   inputs = [prefix + doc for doc in examples["question"]]
   model_inputs = tokenizer(inputs, max_length = 128, truncation = True)
  
   # The "labels" are the tokenized outputs:
   labels = tokenizer(text_target = examples["answer"], 
                      max_length = 256,         
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

BATCH_SIZE = 16
PER_DEVICE_EVAL_BATCH = 16

training_arguments = Seq2SeqTrainingArguments(
    output_dir = "./Results/MedQA/Flan-T5",
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

# %%
# 6. Test and compare the non-fine-tuned model against the fine-tuned MistralAI's model
import tqdm

def generate_test_prompt(x):
    question = '{}\nOptions:\n1. {}\n2. {}\n3. {}\n4. {}\n5. {}\n'.format(x['question'], x['opa'], x['opb'], x['opc'], x['opd'], x['ope'])
    prompt = f"Question:{question}\nAnswer: "
    return prompt
test_df['text'] = test_df.apply(lambda x: generate_test_prompt(x), axis = 1)

# Load the best checkpoint of Mistral-7B-Instruct
model_id = 'Results/MedQA/Flan-T5/checkpoint-1274'
tokenizer = T5Tokenizer.from_pretrained(model_id, use_fast = False)
model = T5ForConditionalGeneration.from_pretrained(
    model_id,
    low_cpu_mem_usage = True,
    return_dict = True,
    use_auth_token = True,
    torch_dtype = torch.bfloat16,
    device_map = "cuda")

generation_config = GenerationConfig(
    do_sample = True,
    top_k = 1,
    temperature = 0.1,
    max_new_tokens = 25,
    pad_token_id = tokenizer.pad_token_id
)

# 7. Load the Test set
def solve_question(question_prompt):
    inputs = tokenizer(question_prompt, return_tensors = "pt", padding = True, truncation = True).to("cuda")
    outputs = model.generate(**inputs, generation_config = generation_config)
    answer = tokenizer.batch_decode(outputs, skip_special_tokens = True)
    return answer

import tqdm
all_answers = []
test_prompts = list(test_df['text'])
for i in tqdm.tqdm(range(0, len(test_prompts), 16)):
    question_prompts = test_prompts[i:i+16]
    ans = solve_question(question_prompts)
    all_answers.extend(ans)


# 8. Score for the accuracy on Test set
pred_answers = []
for i in range(len(test_df)):
    if all_answers[i] == '1.':
        pred_answers.append(test_df['opa'][i])
    elif all_answers[i] == '2.':
        pred_answers.append(test_df['opb'][i])
    elif all_answers[i] == '3.':
        pred_answers.append(test_df['opc'][i])
    elif all_answers[i] == '4.':
        pred_answers.append(test_df['opd'][i])
    elif all_answers[i] == '5.':
        pred_answers.append(test_df['ope'][i])
    else:
        pred_answers.append(all_answers[i])

all_answers = [re.sub(r'\n|"', '', answers) for answers in pred_answers]

correct_answers = []
for i in range(len(test_df)):
    if test_df['answer_idx'][i] == 'A':
        correct_answers.append(test_df['opa'][i])
    elif test_df['answer_idx'][i] == 'B':
        correct_answers.append(test_df['opb'][i])
    elif test_df['answer_idx'][i] == 'C':
        correct_answers.append(test_df['opc'][i])
    elif test_df['answer_idx'][i] == 'D':
        correct_answers.append(test_df['opd'][i])
    elif test_df['answer_idx'][i] == 'E':
        correct_answers.append(test_df['ope'][i])

correct_count = 0
for i in range(len(test_df)):
    correct_count += correct_answers[i] == all_answers[i]
correct_count/len(test_df)


import sacrebleu
import numpy as np
def compute_bleu(preds, labels):
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Prepare references for sacrebleu
    decoded_labels = [[label] for label in decoded_labels]

    # Compute BLEU score
    bleu = sacrebleu.corpus_bleu(decoded_preds, decoded_labels)

    return bleu.score

tokenized_dataset["test"] = tokenized_dataset["test"].remove_columns(['answer_idx', 'opa', 'opb', 'opc', 'opd', 'ope', 'text'])

eval_results = trainer.predict(tokenized_dataset["test"])
preds = eval_results.predictions
labels = eval_results.label_ids

labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

bleu_score = compute_bleu(preds, labels)
print(bleu_score)