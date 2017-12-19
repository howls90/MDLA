import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LinearRegression

boston = datasets.load_boston()

x = boston.data
y = boston.target

boston_df = pd.DataFrame(x, columns=boston.feature_names)


rl = LinearRegression()
rl.fit(x, y)

predicciones = rl.predict(x)


print(np.mean(y - predicciones))