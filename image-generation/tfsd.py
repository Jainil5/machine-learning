from tensorflow import keras
import keras_cv
import cv2
import matplotlib.pyplot as plt

keras.mixed_precision.set_global_policy('mixed_float16')
model = keras_cv.models.StableDiffusion(jit_compile=True)

image = model.text_to_image("man eating stone",batch_size=4)

plt.imshow(image,"Image")