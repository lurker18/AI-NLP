import os
import re
import numpy as np
import pandas as pd

np.set_printoptions(threshold = np.inf, linewidth = np.inf)

path = "./Dataset/Pubmed/"
file_lst = os.listdir(path)
#print(path + file_lst[0])

df = pd.DataFrame()
for file in file_lst:
    filepath = path + file
    data = pd.read_csv(filepath)
    data = data[['Abstract']]
    df = pd.concat([df,data])

df = df.dropna(axis=0)
df = df.reset_index(drop = True)
print(df)

abstract = ''.join(df['Abstract'])

with open('Pubmed_EDA/abstract.txt', 'w') as text_file:
    print(abstract, file = text_file)