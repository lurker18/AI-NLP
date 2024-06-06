import os 
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoModel
import torch
import pandas as pd
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

base_folder = '/media/lurker18/Local Disk/HuggingFace/models/MetaAI/'
embedding_folder = '/media/lurker18/Local Disk/HuggingFace/models/Sentence-Transformers/'
# Define the Prediction Function
def predict_llama3(prompt, system_prompt):
    chat = [
        {"role":"user", "content":f"""You are a helpful chatbot. Use the following information about MedQuAD to answer this question\n" "Do not use any other information. Only base your answer on the given context.\n" "In case you are not sure if your answer is correct with more than 70 percent accuaracy: respond with the following text: I am not sure if I can answer this correctly. Can you please try to rephrase the question?" "Give a complete and well explained answer but provide context within the limits of the information provided here.\n" "Here's the question:\n{prompt}"""},
    ]
    
    if system_prompt == "":
        chat = [
            {"role":"user", "content":f"{prompt}"}
        ]
    prompt = tokenizer.apply_chat_template(chat, tokenize = False, add_generation_prompt = True)
    inputs = tokenizer.encode(prompt, add_special_tokens = False, return_tensors = "pt").to("cuda")
    outputs = model.generate(inputs = inputs, max_new_tokens = 500)
    return tokenizer.decode(outputs[0], skip_special_tokens = True)

# Get the answers seperately from the Prediction
def isolate_answer(input_string):
    # Split the input string by newline characters
    lines = input_string.split("\n")

    # Initialize variabels for storing roles and responses
    current_role = None
    response = ""

    # Iterate over each line
    for line in lines:
        if line == "assistant":
            # If the line contains "model", store the corresponding response
            current_role = "assistant"
            response += line + '\n'
        elif current_role == 'assistant':
            # Accumulate the resposne untial reaching a new role or end of string
            response += line + '\n'
            if 'User:' in line or len(lines) == lines.index(line) + 1:
                lines1 = response.split('\n')
                lines1[0] = ''
                finalresponse = ''
                for r in lines1:
                    finalresponse += r + '\n'
                # Once reached the user's line or end of the string, return the extracted response
                return finalresponse

# 2. Select the tokenizer model
tokenizer = AutoTokenizer.from_pretrained(base_folder + "Llama_3_8B_Instruct", padding = "max_length" , truncation = True)
tokenizer.padding_side = 'right' # to prevent warnings
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_eos_token = True
# 3. Set the quantization settings
bnb_config = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_quant_type = "nf4",
    bnb_4bit_compute_dtype = torch.float16,
    bnb_4bit_use_double_quant = True,
)

# 4. Select the MetaAI's Llama3-8B-chat model
model = AutoModelForCausalLM.from_pretrained(
    base_folder + "Llama_3_8B_Instruct",
    quantization_config = bnb_config,
    attn_implementation = "flash_attention_2",
    torch_dtype = torch.float16,
    device_map = "auto",
    use_auth_token = False,
)
model.config.use_cache = False
model.config.pretraining_tp = 1
model = prepare_model_for_kbit_training(model)
peft_config = LoraConfig(
    lora_alpha = 32,
    lora_dropout = 0.05,
    r = 16,
    bias = "none",
    task_type = "CAUSAL_LM",
    target_modules = ['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj']
)
model = get_peft_model(model, peft_config)

# Test a question
ans = predict_llama3("who are you", "")
print(ans)

# 5. Load the datasets
documents = []
csv_loader = CSVLoader('Dataset/MedQuAD[clean].csv')
documents = csv_loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 100, add_start_index = True)
all_splits = splitter.split_documents(documents)

len(all_splits)

# 6. Load an Embedding Model

model_kwargs = {'device' : 'cuda', 'trust_remote_code' : True}
encode_kwargs = {'normalize_embeddings' : False}
embeddings = HuggingFaceEmbeddings(model_name = embedding_folder + 'all-MiniLM-L6-v2',
                                   model_kwargs = model_kwargs,
                                   encode_kwargs = encode_kwargs)


# 7. Gather and distribute into neat vectorized database for Q&A Preparation
db = FAISS.from_documents(all_splits, embeddings)

test = db.similarity_search('What is (are) Trigeminal Neuralgia')
#print(test[0].page_content)

retriever = db.as_retriever()

def predict_RAG(prompt):
    context = retriever.get_relevant_documents(prompt)
    context_str = ""
    for document in context:
        context_str += document.page_content + '\n'
    return predict_llama3(prompt, context_str)


question = "What is (are) keratoderma with woolly hair?"
answer = predict_RAG(question)
print(answer)