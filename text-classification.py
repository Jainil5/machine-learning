import tensorflow as tf
from tensorflow import keras
import pandas as pd 
import numpy as np

data = keras.datasets.imdb

(x_train,y_test),(y_train,y_test) = data.load_data(num_words=10000)

word_index = keras.datasets.imdb.get_word_index()

print(word_index)