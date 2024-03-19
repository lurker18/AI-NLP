#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 20:24:10 2024

@author: lurker18
"""

## What is RAG?
"""
RAG stands for Retrieval Augmented Generation.
The goal of RAG is to take information and pass it to an LLM so it can generate outputs based on that information.

* Retrieval - Find relevant information given a query, e.g. "What are the macronutrients and what do they do?" --> retrieves passages of text related to the macronutrients from a nutrion textbook.
* Augmented - We want to take the relevant information and augment our input(prompt) to an LLM with that relevant information.
* Generation - Take the first two steps and pass them to an LLM for generative outputs.
"""

# Import PDF Documents
import os
import re
import requests
import torch
import fitz
import textwrap
import random
import numpy as np
import pandas as pd
pd.options.display.max_columns = None
from tqdm import tqdm
from spacy.lang.en import English
from sentence_transformers import util, SentenceTransformer
from time import perf_counter as timer
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoModelForSeq2SeqLM
from transformers.utils import is_flash_attn_2_available

device = "cuda" if torch.cuda.is_available() else "cpu"

# Get PDF document path
pdf_path = "human-nutrition-text.pdf"

# Download PDF
if not os.path.exists(pdf_path):
    print(f"[INFO] File doesn't exist, downloading...")
    
    # Enter the URL of the PDF
    url = "https://pressbooks.oer.hawaii.edu/humannutrition/open/download?type=pdf"
    
    # The local filename to save the downloaded file
    filename = pdf_path
    
    # Send a GET request to the URL
    response = requests.get(url)
    
    # Check if the request was successful
    if response.status_code == 200:
        # Open the file and save it
        with open(filename, "wb") as file:
            file.write(response.content)
        print(f"[INFO] The file has been downloaded and saved as {filename}")
    else:
        print(f"[INFO] Failed to download the file. Status code: {response.status_code}")

else:
    print(f"File {pdf_path} exists.")
    

def text_formatter(text: str) -> str:
    """Performs minor formatting on text."""
    cleaned_text = text.replace("\n", " ").strip()
    
    # Potentially more text formatting functions can go here
    return cleaned_text

def open_and_read_pdf(pdf_path: str) -> list[dict]:
    doc = fitz.open(pdf_path)
    pages_and_texts = []
    for page_number, page in tqdm(enumerate(doc)):
        text = page.get_text()
        text = text_formatter(text = text)
        pages_and_texts.append({"page_number": page_number - 41,
                                "page_char_count": len(text),
                                "page_word_count" : len(text.split(" ")),
                                "page_sentence_count_raw": len(text.split(". ")),
                                "page_token_count" : len(text) / 4, # 1 token = ~4 characters
                                "text" : text})
    return pages_and_texts

pages_and_texts = open_and_read_pdf(pdf_path = pdf_path)
pages_and_texts[:2]

# Read some samples of preprocessed texts
random.sample(pages_and_texts, k = 3)


df = pd.DataFrame(pages_and_texts)
df.head()

df.describe().round(2)

nlp = English()

# Add a sentencizer pipeline
nlp.add_pipe("sentencizer")

# Create document instance as an example
doc = nlp("This is a sentence. This another sentence. I like elephants.")
assert len(list(doc.sents)) == 3

# Print out our sentences split
list(doc.sents)

pages_and_texts[600]
for item in tqdm(pages_and_texts):
    item["sentences"] = list(nlp(item["text"]).sents)
    
    # Make sure all sentences are strings (the default type is a spaCy datatype)
    item["sentences"] = [str(sentence) for sentence in item["sentences"]]
    
    # Count the sentences
    item["page_sentence_count_spacy"] = len(item["sentences"])
    
random.sample(pages_and_texts, k = 1)

df = pd.DataFrame(pages_and_texts)
df.describe().round(2)


### Chunking our sentences together
# Define split size to turn groups of sentences into chunks
num_sentence_chunk_size = 10
 
# Create a function to split lists of texts recursively into chunk size
# e.g. [20] -> [10, 10] or [25] -> [10, 10, 5]
def split_list(input_list: list[str],
               slice_size: int = num_sentence_chunk_size) -> list[list[str]]:
    return [input_list[i : i + slice_size] for i in range(0, len(input_list), slice_size)]

test_list = list(range(25))
split_list(test_list)

# Loop through pages and texts and split sentences into chunks
for item in tqdm(pages_and_texts):
    item["sentence_chunks"] = split_list(input_list = item["sentences"],
                                         slice_size = num_sentence_chunk_size)
    
    item["num_chunks"] = len(item["sentence_chunks"])
    
random.sample(pages_and_texts, k = 1)

df = pd.DataFrame(pages_and_texts)
df.describe().round(2)

### Splitting each chunk into its own item
# Split each chunk into its own item
pages_and_chunks = []
for item in tqdm(pages_and_texts):
    for sentence_chunk in item["sentence_chunks"]:
        chunk_dict = {}
        chunk_dict["page_number"] = item["page_number"]
        
        # Join the sentences together into a paragraph-like structure, aka join the list of sentences into one paragraph
        joined_sentence_chunk = "".join(sentence_chunk).replace("  ", " ").strip()
        joined_sentence_chunk = re.sub(r"\.([A-Z])", r". \1", joined_sentence_chunk) # ".A" => ". A" (will work for any capital letter)
        
        chunk_dict["sentence_chunk"] = joined_sentence_chunk
        
        # Get some stats on our chunks
        chunk_dict["chunk_char_count"] = len(joined_sentence_chunk)
        chunk_dict["chunk_word_count"] = len([word for word in joined_sentence_chunk.split(" ")])
        chunk_dict["chunk_token_count"] = len(joined_sentence_chunk) / 4 # 1 token = ~4 chars
        
        pages_and_chunks.append(chunk_dict)
        
len(pages_and_chunks)

random.sample(pages_and_chunks, k = 1)

df = pd.DataFrame(pages_and_chunks)
df.describe().round(2)


### Filter chunks of text for short chunks
min_token_length = 30
for row in df[df["chunk_token_count"] <= min_token_length].sample(5).iterrows():
    print(f'Chunk token count: {row[1]["chunk_token_count"]} | Text: {row[1]["sentence_chunk"]}')
    
# Filter our DataFrame for rows with under 30 tokens
pages_and_chunks_over_min_token_len = df[df["chunk_token_count"] > min_token_length].to_dict(orient = "records")
pages_and_chunks_over_min_token_len[:2]

random.sample(pages_and_chunks_over_min_token_len, k = 1)

### Embedding our text chunks
embedding_model = SentenceTransformer(model_name_or_path = "all-MiniLM-L12-v2", 
                                      device = device)

# Create a list of sentences
sentences = ["The Sentence Transformer library provides an easy way to create embeddings.",
             "Sentences can be embedded one by one or in a list.",
             "I like horses!"]

# Sentences are encoded/embedded by calling model.encode()
embeddings = embedding_model.encode(sentences)
embeddings_dict = dict(zip(sentences, embeddings))

# See the embeddings
for sentence, embedding in embeddings_dict.items():
    print(f"Sentence: {sentence}")
    print(f"Embedding: {embeddings}")
    print("")
    
    
# Embed each chunk one by one
for item in tqdm(pages_and_chunks_over_min_token_len):
    item["embedding"] = embedding_model.encode(item["sentence_chunk"])
    
text_chunks = [item["sentence_chunk"] for item in pages_and_chunks_over_min_token_len]
text_chunks[419]

len(text_chunks)

# Embed all texts in batches
text_chunk_embeddings = embedding_model.encode(text_chunks,
                                               batch_size = 128, # You can experiment to find which batch size leads to best results
                                               convert_to_tensor = True)
text_chunk_embeddings

### Save embeddings to file
# Save embeddings to file
text_chunks_and_embeddings_df = pd.DataFrame(pages_and_chunks_over_min_token_len)
text_chunks_and_embeddings_df
embeddings_df_save_path = "text_chunks_and_embeddings_df.csv"
text_chunks_and_embeddings_df.to_csv(embeddings_df_save_path, index = False, encoding = 'utf-8-sig')


# Import saved file and view
text_chunks_and_embeddings_df_load = pd.read_csv(embeddings_df_save_path)
text_chunks_and_embeddings_df_load.head()


### 2. RAG - Search and Answer
# RAG goal: Retrieve relevant passages based on a query and use those passages to augment an input to an LLM so it can generate an output based on those relevant passages.

### Similarity search
# Import texts and embedding df
text_chunks_and_embedding_df = pd.read_csv("text_chunks_and_embeddings_df.csv")

# Convert embedding column back to np.array(it got converted to string when it saved to CSV)
text_chunks_and_embedding_df["embedding"] = text_chunks_and_embedding_df["embedding"].apply(lambda x : np.fromstring(x.strip("[]"), sep = " "))

# Convert our embeddings into a torch.tensor
embeddings = torch.tensor(np.stack(text_chunks_and_embedding_df["embedding"].tolist(), axis = 0), dtype = torch.float32).to(device)

# Convert texts and embedding df to list of dicts
pages_and_chunks = text_chunks_and_embeddings_df.to_dict(orient = "records")
pages_and_chunks


# Create model
embedding_model = SentenceTransformer(model_name_or_path = "all-MiniLM-L12-v2", 
                                      device = device)

# 1. Define the query
query = "good foods for protein"
print(f"Query: {query}")

# 2. Embed the query
# Note: it's important to embed your query with the same model you embedded your passages
query_embedding = embedding_model.encode(query, convert_to_tensor = True)

# 3. Get similarity scores with the dot product (use cosine similarity if outputs of model aren't normalized)
#start_time = timer()
dot_scores = util.dot_score(a = query_embedding, b = embeddings)[0]
#end_time = timer()

#print(f"[INFO] Time taken to get scores on {len(embeddings)} embeddings: {end_time - start_time:.5f} seconds. ")

# 4. Get the top-k results (we'll keep top 5)
top_results_dot_product = torch.topk(dot_scores, k = 5)

def print_wrapped(text, wrap_length = 80):
    wrapped_text = textwrap.fill(text, wrap_length)
    print(wrapped_text)
    
    
print(f"Query: '{query}'\n")
print("Results:")
# Loop through zipped together scores and indices from torch.topk
for score, idx in zip(top_results_dot_product[0], top_results_dot_product[1]):
    print(f"Score: {score:.4f}")
    print("Text:")
    print_wrapped(pages_and_chunks[idx]['sentence_chunk'])
    print(f"Page number: {pages_and_chunks[idx]['page_number']}")
    print("\n")
    

# open PDF and load target
pdf_path = "human-nutrition-text.pdf"
doc = fitz.open(pdf_path)
page = doc.load_page(406 + 41) # note: page numbers of our PDF start 41+

# Get the image of the page
img = page.get_pixmap(dpi = 600)

# Save image (optional)
#img.save("output_filename.png")
doc.close()

# Convert the pixmap to a numpy array
img_array = np.frombuffer(img.samples_mv,
                          dtype = np.uint8).reshape((img.h, img.w, img.n))

# Display the image using Matplotlib
plt.figure(figsize = (13, 10))
plt.imshow(img_array)
plt.title(f"Query:'{query}' | Most relevant page:")
plt.axis("off")

# Similarity measures: dot product and cosine similarity
"""
Two of the most common similarity measures between vectors are dot product and cosine similarity. 
In essence, closer vectors will have higher scores, further away vectors will have lower scores.
Vectors have direction (which way is it going?) and magnitude (how long is it?)
"""
def dot_product(vector1, vector2):
    return torch.dot(vector1, vector2)

def cosine_similarity(vector1, vector2):
    dot_product = torch.dot(vector1, vector2)
    
    # Get Euclidean/L2 norm
    norm_vector1 = torch.sqrt(torch.sum(vector1**2))
    norm_vector2 = torch.sqrt(torch.sum(vector2**2))
    
    return dot_product / (norm_vector1 * norm_vector2)

# Example vectors/tensors
vector1 = torch.tensor([1, 2, 3], dtype = torch.float32)
vector2 = torch.tensor([1, 2, 3], dtype = torch.float32)
vector3 = torch.tensor([4, 5, 6], dtype = torch.float32)
vector4 = torch.tensor([-1, -2, -3], dtype = torch.float32)

# Calculate the dot product
print("Dot Product between vector1 and vector2:", dot_product(vector1, vector2))
print("Dot Product between vector1 and vector2:", dot_product(vector1, vector3))
print("Dot Product between vector1 and vector2:", dot_product(vector1, vector4))

# Cosine similarity
print("Cosine similarity between vector1 and vector2:", cosine_similarity(vector1, vector2))
print("Cosine similarity between vector1 and vector3:", cosine_similarity(vector1, vector3))
print("Cosine similarity between vector1 and vector3:", cosine_similarity(vector1, vector4))

### Functionizing our semantic search pipeline (retrieval)
def retrieve_relevant_resources(query: str, 
                                embeddings: torch.tensor,
                                model: SentenceTransformer = embedding_model,
                                n_resources_to_return: int = 10,
                                print_time: bool = True):
    """
    Embeds a query with model and returns top k scores and indices from embeddings.
    """
    
    # Embed the query
    query_embedding = model.encode(query, convert_to_tensor = True)
    
    # Get dot product scores on embeddings
    start_time = timer()
    dot_scores = util.dot_score(query_embedding, embeddings)[0]
    end_time = timer()
    
    if print_time:
        print(f"[INFO] Time taken to get scores on ({len(embeddings)} embeddings: {end_time-start_time:.5f} seconds.")
        
    
    scores, indices = torch.topk(input = dot_scores,
                                 k = n_resources_to_return)
    
    return scores, indices

retrieve_relevant_resources(query = "foods high in fiber", embeddings = embeddings)

def print_top_results_and_scores(query: str,
                                 embeddings: torch.tensor,
                                 pages_and_chunks: list[dict] = pages_and_chunks,
                                 n_resources_to_return: int = 5):
    
    """
    Finds relevant passages given a query and prints them out along with their scores.
    """
    scores, indices = retrieve_relevant_resources(query = query,
                                                  embeddings = embeddings,
                                                  n_resources_to_return = n_resources_to_return)
    
    
    # Loop through zipped together scores and indices from torch.topk
    for score, idx in zip(scores, indices):
        print(f"Score: {score:.4f}")
        print("Text:")
        print_wrapped(pages_and_chunks[idx]['sentence_chunk'])
        print(f"Page number: {pages_and_chunks[idx]['page_number']}")
        print("\n")
        
query = "foods high in fiber"
print_top_results_and_scores(query = query, embeddings = embeddings)


### Getting an LLM for local generation
# Checking our local GPU memory availability
# Get GPU available memory
gpu_memory_bytes = 0
for i in range(torch.cuda.device_count()):
    gpu_memory_bytes += torch.cuda.get_device_properties(i).total_memory
gpu_memory_gb = round(gpu_memory_bytes / (2**30))
print(f"Available GPU memory: {gpu_memory_gb} GB")

### Loading an LLM locally


# 1. Create a quantization configuration
quantization_config = BitsAndBytesConfig(load_in_4bit = True,
                                         bnb_4bit_compute_dtype = torch.float16)


# Bonus: flash attention 2 = faster attention mechanism
# Flash Attention 2 requires a GPU with a compute capability score of 8.0+ (Ampere, Ada Lovelace, Hopper and above)
if (is_flash_attn_2_available()) and (torch.cuda.get_device_capability(0)[0] >= 8):
    attn_implementation = "flash_attention_2"
else:
    attn_implementation = "sdpa" # scaled dot product attention
    
# 2. Pick a model we'd like to use
#model_id = "mistralai/Mistral-7B-Instruct-v0.2" # Mistral 7B
#model_id = "google/gemma-7b-it" # Gemma 7B
model_id = "meta-llama/Llama-2-7b-chat-hf" # Llama 2 7B

# 3. Instantiate tokenizer (tokenizer turns text into tokens)
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path = model_id)

# 4. Instantiate the model
use_quantization_config = True
llm_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path = model_id,
                                                 torch_dtype = torch.float16,
                                                 quantization_config = quantization_config if use_quantization_config else None,
                                                 #low_cpu_mem_usage = False, # use as much memory as we can
                                                 attn_implementation = attn_implementation)
                                                 
if not use_quantization_config:
    llm_model.to(device)
    
    
def get_model_num_params(model: torch.nn.Module):
    return sum([param.numel() for param in model.parameters()])

get_model_num_params(llm_model)

def get_model_mem_size(model: torch.nn.Module):
    # Get model parameters and buffer sizes
    mem_params = sum([param.nelement() * param.element_size() for param in model.parameters()])
    mem_buffers = sum([buf.nelement() * buf.element_size() for buf in model.buffers()])
    
    # Calculate model sizes
    model_mem_bytes = mem_params + mem_buffers
    model_mem_mb = model_mem_bytes / (1024**2)
    model_mem_gb = model_mem_bytes / (1024**3)
    
    return {"model_mem_bytes": model_mem_bytes,
            "model_mem_mb" : round(model_mem_mb, 2),
            "model_mem_gb" : round(model_mem_gb, 2)}

get_model_mem_size(llm_model)

### Generate text with our local LLM
input_text = "What are the macronutrients, and what roles do they play in the human body?"
print(f"Input text:\n{input_text}")

# Create prompt template for instruction-tuned model
dialogue_template = [
    {"role" : "user",
     "content" : input_text}
]

# Apply the chat template
prompt = tokenizer.apply_chat_template(conversation = dialogue_template,
                                       tokenize = False,
                                       add_generation_prompt = True)
print(f"\nPrompt (formatted):\n{prompt}")

# Tokenize the input text (turn it into numbers) and send it to the GPU
input_ids = tokenizer(prompt,
                      return_tensors = "pt").to(device)

# Generate outputs from local LLM
outputs = llm_model.generate(**input_ids,
                             max_new_tokens = 256)

print(f"Model output (tokens):\n{outputs[0]}\n")

# Decode the output tokens to text
outputs_decoded = tokenizer.decode(outputs[0])
print(f"Model output (decoded):\n{outputs_decoded}\n")

