from sklearn.datasets import fetch_california_housing
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression

x, y = fetch_california_housing(return_X_y=True)
model =KNeighborsRegressor()

model.fit(x,y)

print(x[:1],y[:1])
print(model.predict(x))



