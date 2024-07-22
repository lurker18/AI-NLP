import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
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
    output_dir = "./Results/MedQA/opt-2.7b",
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

