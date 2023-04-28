import numpy as np
import pandas as pd
from PIL import Image
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import mnist

x = mnist.train_images()
y = mnist.train_labels()
xt = mnist.test_images()
yt = mnist.test_labels()

print(x.shape,y.shape)
print(xt.shape,yt.shape)

x = x.reshape((-1,28*28))
print(x.shape,y.shape)

xt = xt.reshape((-1,28*28))
print(xt.shape,yt.shape)

print(x[0])

x  = (x/256)
xt = (xt/256)

print(x[0])

model = MLPClassifier(solver="adam",activation="relu",hidden_layer_sizes=(64,64))

model.fit(x,y)

prediction =  model.predict(xt)

acc = accuracy_score(yt,prediction)

print("Pred:",prediction)
print("Actu:",yt)

print("Accuracy:",acc)
