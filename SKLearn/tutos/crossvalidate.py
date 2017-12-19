from sklearn import datasets
from sklearn.model_selection import cross_val_predict
from sklearn import linear_model
import matplotlib.pyplot as plt

lr = linear_model.LinearRegression()
boston = datasets.load_boston()

x = boston.data
y = boston.target

predicted = cross_val_predict(lr, x, y, cv=10)
print predicted

fig, ax = plt.subplots()
ax.scatter(y,predicted,edgecolors=(0,0,0))
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()