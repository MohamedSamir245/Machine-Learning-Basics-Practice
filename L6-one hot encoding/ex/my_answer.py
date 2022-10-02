import pandas as pd
df=pd.read_csv("D:\Machine learning\ml_codebasics\L6-one hot encoding\ex\carprices.csv")

dummmies=pd.get_dummies(df[['Car Model']])

merged=pd.concat([dummmies,df],axis='columns')

final=merged.drop(['Car Model','Car Model_Mercedez Benz C class'],axis='columns')

x=final.drop(['Sell Price($)'],axis='columns').values
y=final[['Sell Price($)']].values

from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x,y)

#print(model.predict([[0,1,86000,7]]))

print(model.score(x,y))

