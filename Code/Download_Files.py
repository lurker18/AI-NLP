from datasets import load_dataset, Dataset
import datasets
import pandas as pd


#### 1. MedQA
dataset = load_dataset("bigbio/med_qa")
train_data = dataset['train']
val_data = dataset['validation']
test_data = dataset['test']

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
    data_hf = Dataset.from_pandas(df)
    return df, data_hf

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
    prompt = f"""Question:
    {question}
    [INST] Solve this medical question-answering and provide the correct option. [/INST]
    Answer: {answer} </s>""" 
    return prompt

train_df, train_hf = convert_format_df(train_data)
val_df, val_hf = convert_format_df(val_data)
test_df, test_hf = convert_format_df(test_data)

dataset2 = datasets.DatasetDict({"train" : train_hf,
                                 "validation" : val_hf,
                                 "test": test_hf})
dataset2

train_df['text'] = train_df.apply(lambda x: generate_prompt(x), axis = 1)
val_df['text'] = val_df.apply(lambda x: generate_prompt(x), axis = 1)
test_df['text'] = test_df.apply(lambda x: generate_prompt(x), axis = 1)
