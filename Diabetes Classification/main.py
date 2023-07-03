import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("Diabetes Classification\diabetes.csv")
X = df[df.columns[:-1]].values
y = df[df.columns[-1]].values
print(df.head())
scaler = StandardScaler()

x_train,x_temp,y_train,y_temp = train_test_split(X,y, test_size=0.4)
x_valid,x_test,y_valid,y_test = train_test_split(x_temp,y_temp,test_size=0.5)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(16,activation="relu"),
    tf.keras.layers.Dense(16,activation="relu"),
    tf.keras.layers.Dense(1,activation="sigmoid")
])

model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.01), 
              loss = tf.keras.losses.BinaryCrossentropy(),metrics=["accuracy"])

model.evaluate(x_train,y_train)

model.evaluate(x_valid,y_valid)

model.fit(x_train,y_train, batch_size=16,epochs=5,validation_data=(x_valid,y_valid))

print(model.predict(x_test[0]))