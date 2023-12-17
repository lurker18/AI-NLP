import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
pd.set_option('display.max_rows', None)

stopwords = set()
with open('stopwords.txt', mode='r', encoding='utf-8') as file:
    for line in file:
        words = line.split()
        stopwords.update(words)

word_frequency = Counter()

with open('nouns.txt') as file:
    for line in file:
        words = line.split()
        for word in words:
            if word not in stopwords:
                word_frequency[word] += 1

num_of_words = 20000
top_words = word_frequency.most_common(num_of_words)

df_top_words = pd.DataFrame(top_words, columns=['words', 'count'])

with open('top_words.txt', 'w') as text_file:
    text_file.write(str(df_top_words))

def visualize_word_freq(data):
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 6))
    sns.barplot(data=data, x='words', y='count')
    plt.xticks(rotation=90)
    plt.xlabel('Words')
    plt.ylabel('Counts')
    plt.title('Word Frequency Histogram')
    plt.tight_layout()
    plt.show()

#visualize_word_freq(df_top_words)
