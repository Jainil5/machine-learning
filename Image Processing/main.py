from keras.datasets import mnist
import numpy as np
import cv2
import matplotlib.pyplot as plt

(xtrain,ytrain),(xtest,ytest) = mnist.load_data()

n = 0
plt.imshow(xtrain[n],cmap="gray")
plt.title(ytrain[n])
plt.show()