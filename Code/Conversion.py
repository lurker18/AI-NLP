import pandas as pd

df = pd.read_json('/home/lurker18/Desktop/Bio_Relation_Extraction/Dataset/Scraped/nhsinform/List_of_Disease_Dictionary.json', lines = True)
df.columns = ['Keyword', 'text']
del df['Keyword']

df.to_json('/home/lurker18/Desktop/Bio_Relation_Extraction/Dataset/Scraped/nhsinform/NHSInform-JSON-Lines.jsonl',orient='records',lines = True)
