import tensorflow as tf
from tensorflow import keras 
import numpy as np
import matplotlib.pyplot as plt

data = keras.datasets.fashion_mnist

(train_img,train_labels),(test_images,test_labels) = data.load_data()

class_names = ["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle Boot"]
	

print(train_img.shape)
print(train_img[0,23,23]) 

train_img = train_img/255.0

test_images = test_images/255.0

model = keras.Sequential([
    keras.layers.Flatten(input_shape = (28,28)),
    keras.layers.Dense(128, activation= "relu"),
    keras.layers.Dense(10,activation = "softmax")
])

model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])

model.fit(train_img,train_labels,epochs = 5)

test_loss , test_acc = model.evaluate(test_images,test_labels)

print("Test accuracy is :", test_acc)

print("Test loss is :", test_loss)

predictions = model.predict(test_images)

print(class_names[np.argmax(predictions[9])])

plt.figure()
plt.imshow(test_images[9])

plt.show()