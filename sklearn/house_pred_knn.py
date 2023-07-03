from sklearn.datasets import fetch_california_housing
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt

x,y = fetch_california_housing(return_X_y=True)
model = KNeighborsRegressor()
print(x[0].shape)
model.fit(x,y)

pred = model.predict(x)

print(pred)
 
# plt.scatter(pred,y)

# plt.show()

plt.show()

plt.show()