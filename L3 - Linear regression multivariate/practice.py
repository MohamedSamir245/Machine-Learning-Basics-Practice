import pandas as pd
import numpy as np
from sklearn import linear_model
import math

df= pd.read_csv("D:\Machine learning\ml_codebasics\L3 - Linear regression multivariate\homeprices.csv")

median_bedrooms=math.floor(df.bedrooms.median())

df.bedrooms=df.bedrooms.fillna(median_bedrooms)

reg=linear_model.LinearRegression()
reg.fit(df[['area','bedrooms','age']],df.price)

print(reg.predict([[3000,3,40]]))

