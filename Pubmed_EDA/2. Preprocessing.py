import os
import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.tag import pos_tag
from collections import Counter
import string
import nltk
import spacy
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

with open('abstract.txt') as f:
    abstract = [doc.strip().split('\t') for doc in f ]
    abstract = [(doc[0]) for doc in abstract if len(doc) == 1]

input_str = str(abstract)
abstract = re.sub(r'\d+', '', input_str)
input_str = abstract
abstract = input_str.strip()
input_str = abstract
translator = str.maketrans('', '', string.punctuation)
abstract = input_str.translate(translator)


def extract_words(text):
    words = word_tokenize(abstract)
    tagged_words = pos_tag(words)
    nouns = [word for word, pos in tagged_words if pos in ['NN', 'NNS', 'NNP', 'NNPS']]
    return nouns

nouns = extract_words(abstract)
print(len(nouns))

nouns_list = []
stopword = stopwords.words('english')
for word in nouns:
    if word not in stopword:
        nouns_list.append(word)


with open('nouns.txt', 'w') as text_file:
    for item in nouns_list:
        text_file.write(f'{item}\n')
