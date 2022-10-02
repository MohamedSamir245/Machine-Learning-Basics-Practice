import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.linear_model import LinearRegression

df=pd.read_csv("D:\Machine learning\ml_codebasics/L1&2 - Linear regression\ex1\canada_per_capita_income.csv")

reg=LinearRegression()
reg.fit(df[['year']],df[['per capita income (US$)']])
plt.xlabel('year',fontsize=15)
plt.ylabel('per capita income (US$)',fontsize=15)
plt.scatter(df.year,df[['per capita income (US$)']],color='red',marker='*')
plt.plot(df.year,reg.predict(df[['year']]),color='blue')
plt.show()


print(reg.predict([[2020]]))

