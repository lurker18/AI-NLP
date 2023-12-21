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
df.fillna


# Add a temporary 'sentence #' column for creating a joined words of sentences
df['Sentence #'] = 'Sentence: 1'
labels_to_ids = {k : v for v, k in enumerate(df.Annotation.unique())}
ids_to_labels = {v : k for v, k in enumerate(df.Annotation.unique())}

df['Word'] = df['Word'].astype('string')
df['Annotation'] = df['Annotation'].astype('string')

#index = df['Annotation'].index[df['Annotation'].apply(pd.isnull)]

# Create a new column called sentence which groups the words by sentence
df['sentence'] = df[['Sentence #', 'Word', 'Annotation']].groupby(['Sentence #'])['Word'].transform(lambda x : ' '.join(x))
# Also, create a new column called 'word_labels' which groups the tags by sentence
df['word_labels'] = df[['Sentence #', 'Word', 'Annotation']].groupby(['Sentence #'])['Annotation'].transform(lambda x : ','.join(x))

df = df[['sentence', 'word_labels']].drop_duplicates().reset_index(drop = True)

df.to_json(data_folder + 'NHSInform-JSON.json', 
           orient = 'records', 
           lines = True)
df.head()

len(df)
