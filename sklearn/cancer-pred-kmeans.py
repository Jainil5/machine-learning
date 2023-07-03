# Breast cancer prediction with kMeans clutering algorithm

from sklearn.datasets import load_breast_cancer
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import scale
import pandas as pd

bc = load_breast_cancer()

x = scale(bc.data)
y = bc.target

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2) 

print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)

model = KMeans(n_clusters=2,random_state=0)

model.fit(x_train)

pred = model.predict(x_test)

labels = model.labels_
print("labels:",labels)

print(pred)
print(accuracy_score(y_test,pred))