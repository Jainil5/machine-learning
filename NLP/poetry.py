import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
#from keras.preprocessing.sequence import pad_sequences

data = "In the town of Athy one Jeremy Lanigan \n Battered away ... ..."
corpus = data.lower().split("\n")

tokenizer = Tokenizer(num_words=100,oov_token="<OOV>")
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index)
word_index = tokenizer.word_index #Gives index for each word
#print(word_index)
total_words = len(tokenizer.word_index) + 1
print(total_words)

input_sequences = []

for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1,len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)
        
print(input_sequences)
#Test

# test = ["i really love my dog","my dog loves my manatee"]
# test_seq = tokenizer.texts_to_sequences(test)
# print(test_seq)