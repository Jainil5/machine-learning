from sklearn.datasets import fetch_20newsgroups
import nltk
import numpy as np
from nltk.tokenize import sent_tokenize,word_tokenize


text_data = fetch_20newsgroups()

#type(text_data)    // Bunch

raw_text = text_data.data[:4]

clean_text = []

for i in raw_text:
    clean_text.append(str(i).lower())

#print(clean_text)


#    TOkenizer

sent_tok = []

for sent in clean_text:
    sent = sent_tokenize(sent)
    sent_tok.append(sent)

  

