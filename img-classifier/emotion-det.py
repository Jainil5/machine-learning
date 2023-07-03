import tensorflow as tf
import os
import cv2
import imghdr
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd

path = "img-classifier/"
data_dir = "img-classifier/data"
img_exts = ["jpeg","jpg","bmp","png"]

# for img_class in os.listdir(data_dir):
#     for image in os.listdir(os.path.join(data_dir,img_class)):
#         image_path = os.path.join(data_dir,img_class,image)
        
#         try:
#             img = cv2.imread(image_path)
#             tip = imghdr.what(image_path)

#             if tip not in img_exts:
#                 os.remove(image_path)
#         except Exception as e:
#             print("issue with image:{}".format(image_path))

data = tf.keras.utils.image_dataset_from_directory(os.path.join(path,"data"))
data_iterator = data.as_numpy_iterator() #used becuz we cannot itereate shrough data
batch = data_iterator.next() #with batch size it will load new everytime
#Class 1 = sad ppl and 2 is happy

scaled = batch[0]/255

data = data.map(lambda x,y:(x/255,y))
scaled_iterator = data.as_numpy_iterator().next()
scaled_iterator.next()

