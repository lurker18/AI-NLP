from datasets import load_dataset, Dataset
import datasets
import pandas as pd

# Defining changing the formats for each datasets
def convert_format_df(data, data_name = 'medqa'):
    # MedQA Dataset
    if data_name == 'medqa':
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
    # MedMCQA Dataset
    elif data_name == 'medmcqa':
        data_extracted = [
        {'question' : data['question'][i],
        'cop' : data['cop'][i],
        'opa' : data['opa'][i],
        'opb' : data['opb'][i],
        'opc' : data['opc'][i],
        'opd' : data['opd'][i],
        }
        for i in range(len(data))
        ]
    # PubMedQA Dataset
    elif data_name == 'pubmedqa':
        data_extracted = [
        {'question' : data['QUESTION'][i], 
        'context' : data['CONTEXTS'][i],
        'answer' : data['LONG_ANSWER'][i],
        'answer_idx' : data['final_decision'][i]
        }
        for i in range(len(data))
        ]
    # MMLU Dataset
    elif data_name == 'mmlu':
        data_extracted = [
        {'question' : data['question'][i], 
        'subject' : data['subject'][i], 
        'answer' : data['answer'][i],
        'opa' : data['choices'][i][0], 
        'opb' : data['choices'][i][1], 
        'opc' : data['choices'][i][2], 
        'opd' : data['choices'][i][3]}
        for i in range(len(data))
        ]
    df = pd.DataFrame(data_extracted)
    data_hf = Dataset.from_pandas(df)
    return df, data_hf

# Defining generating instruction prompts for each datasets
def generate_prompt(x, data_name = 'medqa'):
    # MedQA Dataset
    if data_name == 'medqa':
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
    
    # MedMCQA Dataset
    elif data_name == 'medmcqa':
        cop = 'Nothing'
        if x['cop'] == 1:
            cop = x['opa']
        elif x['cop'] == 2:
            cop = x['opb']
        elif x['cop'] == 3:
            cop = x['opc']
        elif x['cop'] == 4:
            cop = x['opd']
        question = '{}\nOptions:\n1. {}\n2. {}\n3. {}\n4. {}\n'.format(x['question'], x['opa'], x['opb'], x['opc'], x['opd'])
        answer = cop
        prompt = f"""
        Question:
        {question}
        [INST] Solve this post graduate medical entrance exam MCQ and provide the correct option. [/INST]
        Answer: {answer} </s>""" 
    
    # PubMedQA Dataset
    elif data_name == 'pubmedqa':
        prompt = f"<s>[INST] <<SYS>> You are an expert in medicine, genetics, and human biology. <</SYS>> Here is my question: {x['question']} Context: {' Context: '.join(x['context'])} Think step by step to answer my question. [/INST] {x['answer']} </s>"

    # MMLU Dataset
    elif data_name == 'mmlu':
        answer_idx = 'Nothing'
        if x['answer'] == 0:
            answer_idx = x['opa']
        elif x['answer'] == 1:
            answer_idx = x['opb']
        elif x['answer'] == 2:
            answer_idx = x['opc']
        elif x['answer'] == 3:
            answer_idx = x['opd']
        question = '{}\nOptions:\n1. {}\n2. {}\n3. {}\n4. {}\n'.format(x['question'], 
                                                                    x['opa'],
                                                                    x['opb'], 
                                                                    x['opc'], 
                                                                    x['opd'])
        answer = answer_idx
        prompt = f"""Question:
        {question}
        [INST] Solve this medical and biological question-answering and provide the correct option. [/INST]
        Answer: {answer} </s>""" 
    return prompt

# Defining getting mmlu datasets
def get_mmlu_datasets():
    clinical_knowledge = load_dataset("cais/mmlu", 'clinical_knowledge')
    medical_genetics = load_dataset("cais/mmlu", 'medical_genetics')
    anatomy = load_dataset("cais/mmlu", 'anatomy')
    pro_medicine = load_dataset("cais/mmlu", 'professional_medicine')
    college_biology = load_dataset("cais/mmlu", 'college_biology')
    college_medicine = load_dataset("cais/mmlu", 'college_medicine')
    print("### 1. Loading all MMLU Subsets.....Complete")

    dataset = datasets.DatasetDict({"clinical_knowledge" : clinical_knowledge,
                                    "medical_genetics" : medical_genetics,
                                    "anatomy" : anatomy,
                                    "professional_medicine": pro_medicine,
                                    "college_biology" : college_biology,
                                    "college_medicine" : college_medicine
                                    })

    dataset['clinical_knowledge'].set_format(type = 'pandas')
    dataset['medical_genetics'].set_format(type = 'pandas')
    dataset['anatomy'].set_format(type = 'pandas')
    dataset['professional_medicine'].set_format(type = 'pandas')
    dataset['college_biology'].set_format(type = 'pandas')
    dataset['college_medicine'].set_format(type = 'pandas')
    print("### 2. Changing Data Format to Pandas.....Complete")

    ck = dataset['clinical_knowledge']['test'][:]
    mg = dataset['medical_genetics']['test'][:]
    an = dataset['anatomy']['test'][:]
    pm = dataset['professional_medicine']['test'][:]
    cb = dataset['college_biology']['test'][:]
    cm = dataset['college_medicine']['test'][:]

    merged_df = pd.concat([ck, mg, an, pm, cb, cm]).reset_index()
    del merged_df['index']
    print("### 3. Concatenating Data into a single DataFrame.....Complete")
    return merged_df


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