# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 23:35:25 2023

@author: Hydra18
"""

import re
import pandas as pd

# New annotation string to add
NEW_PREFIX_STRING = ' [disease]'

class main:
    # Set Initiate: Search for the file directory
    def __init__(self, file):
        super().__init__()
        self.file = file
    # end_def
    
    # Read the file
    def read_file(self):
        if self.file.endswith((".csv")):
            df = pd.read_csv(self.file, encoding = 'utf-8')
        elif self.file.endswith(('.json')):
            df = pd.read_json(self.file, lines = True)
        elif self.file.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(self.file)
        else:
            print("Error - Wrong file read. Either CSV/JSON/XLSX files are accepted!")
        
        # Rename column names
        df.columns = ['Keyword', 'Definition']
        
        # Text Preprocessing
        df['Keyword'] = df['Keyword'].apply(lambda x : x.lower())
        df['Definition'] = df['Definition'].apply(lambda x : x.lower())
        
        # Iterate over each keywords in the DataFrame Definition column
        for index, row in df.iterrows():
            # Apply the `NEW_PREFIX_STRING` corresponding each keyword from the df Keyword column.
            df.at[index, 'Definition'] = re.sub(re.escape(row['Keyword']), lambda match: match.group() + NEW_PREFIX_STRING, row['Definition'])
            
        return df
    # end_def

if __name__ == "__main__":
    temp = main(input("Insert the file directory of the JSON/CSV/XLSX file location: "))
    df = temp.read_file()
    df.to_json(input("Insert the save file location directory (DEFAULT: .JSON): ") + '.json', orient = 'records', lines = True)