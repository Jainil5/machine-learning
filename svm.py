from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
import numpy as np
from sklearn import svm
from sklearn.metrics import *



iris = datasets.load_iris()

x = iris.data
y = iris.target

classes = ["Iris Setosa","Iris Versicolour","Iris Virginica"]



x_train,x_test,y_train,y_test =train_test_split(x,y,test_size=0.2)

print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)

model = svm.SVC()
model.fit(x_train,y_train)

print(model)

predictions = model.predict(x_test)
accuracy = accuracy_score(y_test,predictions)

print("predictions",predictions)
print("acctual:   ",y_test)
print(accuracy)

for i in range(len(predictions)):
    print(i,classes[predictions[i]])