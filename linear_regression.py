from sklearn import datasets
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

boston = datasets.load_digits()

X = boston.data
y = boston.target

print(X.shape,y.shape)