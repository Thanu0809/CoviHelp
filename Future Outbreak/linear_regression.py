# -*- coding: utf-8 -*-
"""Linear_Regression.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Aw_Z7rh1fDZSHgXWi0w37nisly-21IIQ
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
import seaborn as sns

# Commented out IPython magic to ensure Python compatibility.
from google.colab import drive
drive.mount('/content/drive')
# %cd '/content/drive/My Drive'

import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv("COVID-19_Daily_Testing.csv")
data.head()

data['Cases'] = data['Cases'].str.replace(',', '')
data['Tests'] = data['Tests'].str.replace(',', '')

data['Cases'] = pd.to_numeric(data['Cases'])
data['Tests'] = pd.to_numeric(data['Tests'])

plt.figure(figsize=(16, 8))
plt.scatter(
    data['Tests'],
    data['Cases'],    
    c='black'
)
plt.axis('scaled')
plt.xlabel("Tests")
plt.ylabel("Cases")
plt.show()

X = data['Tests'].values.reshape(-1,1)
y = data['Cases'].values.reshape(-1,1)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
reg = LinearRegression()
reg.fit(X, y)
print("The linear model is: Y = {:.5} + {:.5}X".format(reg.intercept_[0], reg.coef_[0][0]))
predictions = reg.predict(X)

actvspred = pd.DataFrame({'Actual': y.flatten(), 'Predicted': predictions.flatten()})

plt.figure(figsize=(16, 8))
plt.scatter(
    X,
    y,
    c='black'
)
plt.plot(
    X,
    predictions,
    c='blue',
    linewidth=2
)
plt.xlabel("Tests")
plt.ylabel("Cases")
plt.show()

print('RMSE for Linear Regression=>',np.sqrt(mean_squared_error(y,predictions)))

poly = PolynomialFeatures(degree =4) 
X_poly = poly.fit_transform(X)

poly.fit(X_poly, y) 
lin2 = LinearRegression() 
lin2.fit(X_poly, y) 
pred = lin2.predict(X_poly)
new_X, new_y = zip(*sorted(zip(X, pred)))

plt.figure(figsize=(16, 8))
plt.scatter(
    X,
    y,
    c='black'
)
plt.plot(
    new_X, new_y,
    c='blue'
)
plt.xlabel("Tests")
plt.ylabel("Cases")
plt.show()
print('RMSE for Linear Regression=>',np.sqrt(mean_squared_error(y,lin2.predict(poly.fit_transform(X)))))