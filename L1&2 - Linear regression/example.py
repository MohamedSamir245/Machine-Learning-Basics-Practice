import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.linear_model import LinearRegression

df= pd.read_csv("D:\Machine learning\ml_codebasics/L1&2 - Linear regression\homeprices.csv")





reg=LinearRegression()


reg.fit(df[['area']],df[['price']])
plt.xlabel('area',fontsize=20)
plt.ylabel('price',fontsize=20)
plt.scatter(df.area,df.price,color='red',marker='*')
plt.plot(df.area,reg.predict(df[['area']]),color='blue')
plt.show()



d=pd.read_csv("D:\Machine learning\pd.csv")

p=reg.predict(d)

d['prices']=p
d.to_csv("D:\Machine learning\prediction.csv",index=False)
