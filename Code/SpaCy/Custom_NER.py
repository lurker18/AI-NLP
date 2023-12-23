import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertConfig, BertForTokenClassification
from torch import cuda

device = 'cuda' if cuda.is_available() else 'cpu'

data_folder = './Dataset/Scraped/nhsinform/'

df = pd.read_csv(data_folder + 'nhsinform-IOB - Modified - Copy.txt', 
                 sep = '\t', 
                 header = None,
                 names = ['Word', 'Annotation']
                 )

df['number'] = 0
sentence_number = 1
for index, row in df.iterrows():
    if row['Word'] == '###':
        df.at[index, 'number'] = sentence_number
        sentence_number += 1
    else:
        df.at[index, 'number'] = sentence_number

# Add a temporary 'sentence #' column for creating a joined words of sentences
df['Sentence #'] = 'TEMP'
labels_to_ids = {k : v for v, k in enumerate(df.Annotation.unique())}
ids_to_labels = {v : k for v, k in enumerate(df.Annotation.unique())}

df['Word'] = df['Word'].astype('string')
df['Annotation'] = df['Annotation'].astype('string')

#index = df['Annotation'].index[df['Annotation'].apply(pd.isnull)]

# Create a new column called sentence which groups the words by sentence
df['sentence'] = df[['Sentence #', 'Word', 'Annotation', 'number']].groupby(['number'])['Word'].transform(lambda x : ' '.join(x))
# Also, create a new column called 'word_labels' which groups the tags by sentence
df['word_labels'] = df[['Sentence #', 'Word', 'Annotation', 'number']].groupby(['number'])['Annotation'].transform(lambda x : ','.join(x))

df = df[['sentence', 'word_labels']].drop_duplicates().reset_index(drop = True)

df['sentence'] = df['sentence'].apply(lambda x: x.replace(' ###', ''))
df['word_labels'] = df['word_labels'].apply(lambda x: x.replace(',###', ''))


df.to_json(data_folder + 'NHSInform-JSON.json', 
           orient = 'records', 
           lines = True)
df.head()

len(df)
