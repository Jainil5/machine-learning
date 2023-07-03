# iris classification using svm

from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()

x,y = iris.data,iris.target

classes = ["Iris Setosa","Iris Versicolor","Iris virginica"]
print(x.shape,y.shape)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

#print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)

model = SVC()

model.fit(x_train,y_train)


pred = model.predict(x_test)

acc = accuracy_score(y_test,pred)

print("Predication:",pred)
print("Acctual    :",y_test)
print("acc:",acc)


for i in range(len(pred)):
    print(classes[pred[i]])