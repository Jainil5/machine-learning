import pandas as pd
from sklearn import datasets
from sklearn.cluster import KMeans
from  sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.metrics import accuracy_score

bc = datasets.load_breast_cancer()
x = scale(bc.data)
#print(x)
y = bc.target
#print(y)

# 1 is train and 2 is test
x1,x2,y1,y2 = train_test_split(x,y,test_size=0.2)

model = KMeans(n_clusters=2,random_state=0)

model.fit(x1) # it is clustering so we dont apply y train with it

pred = model.predict(x2)

labels = model.labels_

print("Labels:",labels)
print("pred:",pred)
print("acc :",y2)
print("acc score:",accuracy_score(y2,pred))
print(pd.crosstab(y1,labels))