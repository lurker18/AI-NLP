from Code.utils import convert_format_df, get_mmlu_datasets, generate_prompt
from datasets import load_dataset, Dataset
import datasets
import pandas as pd


#### 1. MedQA
dataset = load_dataset("bigbio/med_qa")
train_data = dataset['train']
val_data = dataset['validation']
test_data = dataset['test']

train_df, train_hf = convert_format_df(train_data,  data_name = 'medqa')
val_df, val_hf = convert_format_df(val_data,  data_name = 'medqa')
test_df, test_hf = convert_format_df(test_data,  data_name = 'medqa')

dataset2 = datasets.DatasetDict({"train" : train_hf,
                                 "validation" : val_hf,
                                 "test": test_hf})

train_df['text'] = train_df.apply(lambda x: generate_prompt(x, data_name = 'medqa'), axis = 1)
val_df['text'] = val_df.apply(lambda x: generate_prompt(x, data_name = 'medqa'), axis = 1)
test_df['text'] = test_df.apply(lambda x: generate_prompt(x, data_name = 'medqa'), axis = 1)

#### 2. MedMCQA
dataset = load_dataset("openlifescienceai/medmcqa")
dataset.set_format(type = 'pandas')
train_data = dataset['train'][:]
val_data = dataset['validation'][:]
test_data = dataset['test'][:]

train_df, train_hf = convert_format_df(train_data, data_name = 'medmcqa')
val_df, val_hf = convert_format_df(val_data, data_name = 'medmcqa')
test_df, test_hf = convert_format_df(test_data, data_name = 'medmcqa')

dataset2 = datasets.DatasetDict({"train" : train_hf,
                                 "validation" : val_hf,
                                 "test": test_hf})

train_df['text'] = train_df.apply(lambda x: generate_prompt(x, data_name = 'medmcqa'), axis = 1)
val_df['text'] = val_df.apply(lambda x: generate_prompt(x, data_name = 'medmcqa'), axis = 1)
test_df['text'] = test_df.apply(lambda x: generate_prompt(x, data_name = 'medmcqa'), axis = 1)


#### 3. PubMedQA
"""
PubMedQA has 1k expert-annotated (PQA-L), 61.2k unlabeled (PQA-U) and 211.3k artificially generated QA instances (PQA-A).
"""
dataset = load_dataset("bigbio/pubmed_qa")
dataset.set_format(type = 'pandas')
train_data = dataset['train'][:]
val_data = dataset['validation'][:]

train_df, train_hf = convert_format_df(train_data, data_name = 'pubmedqa')
val_df, val_hf = convert_format_df(val_data, data_name = 'pubmedqa')

dataset2 = datasets.DatasetDict({"train" : train_hf,
                                 "validation" : val_hf})

train_df['text'] = train_df.apply(lambda x: generate_prompt(x, data_name = 'pubmedqa'), axis = 1)
val_df['text'] = val_df.apply(lambda x: generate_prompt(x, data_name = 'pubmedqa'), axis = 1)

#### 4. MMLU for Benchmark Evaluation (Not suited for Fine-Tuning. Only Evaluation Comparison)
test_data = get_mmlu_datasets()

test_df, test_hf = convert_format_df(test_data, data_name = 'mmlu')
test_df['text'] = test_df.apply(lambda x : generate_prompt(x, data_name = 'mmlu'), axis = 1)