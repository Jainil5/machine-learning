from tensorflow import keras
import keras_cv
import cv2

keras.mixed_precision.set_global_policy("mixed_float16")
model = keras_cv.models.StableDiffusion(jit_compile=True)

images = model.text_to_image("Teddy bears",batch_size = 4)

cv2.imshow(images)

cv2.waitKey(0)