# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 12:41:01 2023

@author: Hydra18
"""

import re
import pandas as pd
import spacy

file = 'F:/BioNER-Abbrev/Dataset/Scraped/nhsinform/List_of_Disease_Dictionary.json'
def main(file):
    if file.endswith((".csv")):
        df = pd.read_csv(file, encoding = 'utf-8')
    elif file.endswith(('.json')):
        df = pd.read_json(file, lines = True)
    elif file.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(file)
    else:
        print("Error - Wrong file read. Either CSV/JSON/XLSX files are accepted!")
        
    df.columns = ['Keyword', 'Definition']
    
    # Load the spaCy model
    nlp = spacy.load("en_core_web_sm")
    
    # Initialize an empty list to store tagged words
    tagged_words = []
    for f, g in zip(df['Keyword'], df['Definition']):
        print(f"Keyword: {f}")
        keyword = f
        sentence = g
        
        # Tokenize the keyword
        keyword_tokens = keyword.split()
        
        # Process the sentence using spaCy
        doc = nlp(sentence)
        
        # Flag to determine if we are inside the keyword
        inside_keyword = False
        
        # Iterate through each word in the processed sentence
        for token in doc:
            # Check if the current token is part of the keyword
            if token.text.lower() == keyword_tokens[0].lower():
                # Mark the beginning of the keyword
                inside_keyword = True
                tag = 'B-Disease'
                # If the keyword has only one token, mark it and reset the flag
                if len(keyword_tokens) == 1:
                    inside_keyword = False
    
            elif inside_keyword or token.text.lower() == keyword_tokens[-1].lower():
                # If inside the keyword and it's the last word, mark as 'I-Disease'
                tag = 'I-Disease'
                inside_keyword = False
            elif inside_keyword:
                # If inside the keyword and not the last word, mark subsequent words with 'I-Disease'
                tag = 'I-Disease'
            else:
                # If outside the keyword, use 'O' tag
                tag = 'O'
    
            # Append the word and its tag to the list
            tagged_words.append((token.text, tag))

    return tagged_words

temp = main(file)
df = pd.DataFrame(temp)
df.to_csv('F:/BioNER-Abbrev/Dataset/Scraped/nhsinform/nhsinform-IOB.txt', sep = '\t', index = False)
