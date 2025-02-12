import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import re
import pandas as pd
import warnings
import torch
from datasets import load_dataset, Dataset
from peft import LoraConfig, AutoPeftModelForCausalLM, prepare_model_for_kbit_training, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer, TrainingArguments, GenerationConfig)
import tqdm
from tqdm import tqdm
import tensorrt as trt
from trl import SFTTrainer
from Code.utils import convert_format_df

os.environ["WANDB_DISABLED"] = "true"
warnings.filterwarnings("ignore")

base_folder = "/mnt/nvme01/huggingface/models/MistralAI/"

# 1. Load the Dataset
dataset = load_dataset("bigbio/pubmed_qa")
dataset.set_format(type = 'pandas')
train_data = dataset['train'][:]
val_data = dataset['validation'][:]

# 2. Preset the the Instruction-based prompt template
train_df, train_hf = convert_format_df(train_data, data_name = 'pubmedqa')
val_df, val_hf = convert_format_df(val_data, data_name = 'pubmedqa')

# 3. Set the quantization settings
bnb_config = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_quant_type = "nf4",
    bnb_4bit_compute_dtype = torch.bfloat16,
    bnb_4bit_use_double_quant = False,
)

# 4. Select the MistralAI's Mistral-7B-Instruct model
model = AutoModelForCausalLM.from_pretrained(
    base_folder + "Mistral-7B-Instruct-v0.2",
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
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head",]
)
model = get_peft_model(model, peft_config)

# 4.1 Select the tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_folder + "Mistral-7B-Instruct-v0.2", padding = "max_length", truncation = True, max_length = 2048)
tokenizer.pad_token = tokenizer.unk_token
tokenizer.pad_token_id =  tokenizer.unk_token_id
tokenizer.padding_side = 'left' # to prevent warnings

training_arguments = TrainingArguments(
    output_dir = "./Results/PubMedQA/Mistral-7B-Instruct",
    num_train_epochs = 5,
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

# # 6. Test and compare the non-fine-tuned model against the fine-tuned MistralAI's model
# def generate_test_prompt(x):
#     prompt = f"<s>[INST] <<SYS>> You are an expert in medicine, genetics, and human biology. <</SYS>> Here is my question: {x['question']} Context: {' Context: '.join(x['context'])} Think step by step to answer my question. [/INST] {x['answer']} </s>"
#     return prompt
# test_df['text'] = test_df.apply(lambda x: generate_test_prompt(x), axis = 1)

# # Load the best checkpoint of Mistral-7B-Instruct
# model_id = 'Results\PubMedQA\Mistral-7B-Instruct\checkpoint-3185'
# tokenizer = AutoTokenizer.from_pretrained(model_id)
# tokenizer.pad_token = tokenizer.unk_token
# tokenizer.pad_token_id =  tokenizer.unk_token_id
# tokenizer.padding_side = 'left' # to prevent warnings
# model = AutoPeftModelForCausalLM.from_pretrained(
#     model_id,
#     low_cpu_mem_usage = True,
#     return_dict = True,
#     torch_dtype = torch.bfloat16,
#     device_map = "cuda")

# generation_config = GenerationConfig(
#     do_sample = True,
#     top_k = 1,
#     temperature = 0.1,
#     max_new_tokens = 25,
#     pad_token_id = tokenizer.pad_token_id
# )

# # 7. Load the Test set
# def solve_question(question_prompt):
#     inputs = tokenizer(question_prompt, return_tensors = "pt", padding = True, truncation = True).to("cuda")
#     outputs = model.generate(**inputs, generation_config = generation_config)
#     answer = tokenizer.batch_decode(outputs, skip_special_tokens = True)
#     return answer

# all_answers = []
# test_prompts = list(test_df['text'])
# for i in tqdm.tqdm(range(0, len(test_prompts), 16)):
#     question_prompts = test_prompts[i:i+16]
#     ans = solve_question(question_prompts)
#     ans_option = []
#     for text in ans:
#         ans_option.append(re.search(r'Answer: \s*(.*)', text).group(1))
#     all_answers.extend(ans_option)

# all_answers_1 = [re.sub(r'</s>|://|</s|</|s>|s/|.swing', '', answers) for answers in all_answers]
# url_pattern = r'\b\S*\.com\S*|\b\S*\.gov\S*|\b\S*\.org\S*|\b\S*\.jpg'
# all_answers_2 = [re.sub(url_pattern, '', answers) for answers in all_answers_1]
# all_answers_3 = [answers.strip() for answers in all_answers_2]
# all_answers_3


# # 8. Score for the accuracy on Test set


# correct_count = 0
# for i in range(len(test_df)):
#     correct_count += correct_answers[i] == all_answers[i]
# correct_count/len(test_df)