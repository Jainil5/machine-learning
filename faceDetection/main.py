import os
import tensorflow as tf
import cv2
import json
import numpy
import matplotlib.pyplot as plt

images = tf.data.Dataset.list_files("data\images\*.jpg",shuffle=False)
images.as_numpy_iterator().next()
def load_image(x):
    byte_img = tf.io.read_file(x)
    img = tf.io.decode_jpeg(byte_img)
    return img

images = images.map(load_image)

images.as_numpy_iterator().next()

image_generator = images.batch(4).as_numpy_iterator()

plot_images = image_generator.next()

fig, ax = plt.subplots(ncols = 4 , figsize = (20,20))
for idx, image in enumerate(plot_images):
    ax[idx].imshow(image)
plt.show()    