from statistics import linear_regression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

dataset=pd.read_csv("D:\Machine learning\ml_codebasics\L16-regularization\Melbourne_housing_FULL.csv")
# print(dataset.head())

# print(dataset.shape)

cols_to_use = ['Suburb', 'Rooms', 'Type', 'Method', 'SellerG', 'Regionname', 'Propertycount', 
               'Distance', 'CouncilArea', 'Bedroom2', 'Bathroom', 'Car', 'Landsize', 'BuildingArea', 'Price']
dataset = dataset[cols_to_use]

# print(dataset.isna().sum())

cols_to_fill_zero = ['Propertycount', 'Distance', 'Bedroom2', 'Bathroom', 'Car']
dataset[cols_to_fill_zero] = dataset[cols_to_fill_zero].fillna(0)

dataset['Landsize'] = dataset['Landsize'].fillna(dataset.Landsize.mean())
dataset['BuildingArea'] = dataset['BuildingArea'].fillna(dataset.BuildingArea.mean())

dataset.dropna(inplace=True)
# print(dataset.isna().sum())

dataset=pd.get_dummies(dataset,drop_first=True)

x=dataset.drop('Price',axis=1)
y=dataset['Price']

from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(x, y, test_size=0.3, random_state=2)

# from sklearn.linear_model import LinearRegression
# reg=LinearRegression()
# reg.fit(train_X,train_y)
# print(reg.score(test_X,test_y))
# print(reg.score(train_X,train_y))

# from sklearn import linear_model
# lasso_reg=linear_model.Lasso(alpha=50,max_iter=100,tol=0.1)
# lasso_reg.fit(train_X,train_y)
# print(lasso_reg.score(test_X,test_y))

from sklearn.linear_model import Ridge
ridge_reg= Ridge(alpha=50,max_iter=100,tol=0.1)
ridge_reg.fit(train_X, train_y)

print(ridge_reg.score(test_X, test_y))
print(ridge_reg.score(train_X, train_y))

