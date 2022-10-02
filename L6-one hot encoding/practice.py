from statistics import linear_regression
import pandas as pd 
df=pd.read_csv("D:\Machine learning\ml_codebasics\L6-one hot encoding\homeprices.csv")

# dummies=pd.get_dummies(df.town)

# merged=pd.concat([df,dummies],axis='columns')

# final=merged.drop(['town','west windsor'],axis='columns')

# x=final.drop(['price'],axis='columns')

# y=final.price

from sklearn.linear_model import LinearRegression
model=LinearRegression()
# model.fit(x,y)
# print(model.predict([[3400,0,0]]))

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
dfle=df
dfle.town=le.fit_transform(dfle.town)

x=dfle[['town','area']].values

y=dfle.price.values

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct=ColumnTransformer([('towm',OneHotEncoder(),[0])],remainder='passthrough')
x=ct.fit_transform(x)

x=x[:,1:]

model.fit(x,y)

print(model.predict([[0,1,3400]]))

