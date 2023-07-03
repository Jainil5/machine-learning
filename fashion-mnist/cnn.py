# Fashion MNIST using PyTorch

import torch
from  torch import _nnpack_available
import torchvision
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

train_data = datasets.FashionMNIST(root="data",train=True,download=True,transform=ToTensor(),target_transform=None)
test_data = datasets.FashionMNIST(root="data",train=False,download=True,transform=ToTensor(),target_transform=None)

# print(len(train_data),len(test_data)) //60000,10000
image,label = train_data[0]

class_names = train_data.classes
#print(class_names) ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
#print(train_data.class_to_idx) {'T-shirt/top': 0, 'Trouser': 1, 'Pullover': 2, 'Dress': 3, 'Coat': 4, 'Sandal': 5, 'Shirt': 6, 'Sneaker': 7, 'Bag': 8, 'Ankle boot': 9}
