# GAN with tensorflow on fashion mnsit dataset
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, Reshape, LeakyReLU, Dropout, UpSampling2D

ds = tfds.load("fashion_mnist", split="train")
data_iterator = ds.as_numpy_iterator()
data_iterator.next()


def scale_images(data):
    image = data["image"]
    return image / 255


ds = tfds.load("fashion_mnist", split="train")
ds = ds.map(scale_images)
ds = ds.cache()
ds = ds.shuffle(60000)
ds = ds.batch(128)
ds = ds.prefetch(64)


def build_generator():
    model = Sequential()

    model.add(Dense(7 * 7 * 128, input_dim=128))
    model.add(LeakyReLU(0.2))
    model.add(Reshape((7, 7, 128)))

    model.add(UpSampling2D)
    model.add(Conv2D(128, 5, padding="same"))
    model.add(LeakyReLU(0.2))

    return model


test_model = build_generator()
print(test_model.summary())
