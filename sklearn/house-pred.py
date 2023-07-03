# House prediction using linear regression

from sklearn.datasets import fetch_california_housing 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X,y = fetch_california_housing(return_X_y=True)
# print(x.shape,y.shape)

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

model = LinearRegression()

model.fit(x_train,y_train)

pred = model.predict(x_test)

print("pred:",pred)

print("actu:",y_test)

acc = accuracy_score(pred,y_test)
print(acc)


