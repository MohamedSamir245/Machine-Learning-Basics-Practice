import pandas as pd
df=pd.read_csv("D:\Machine learning\ml_codebasics\L7-train test split\carprices.csv")

import matplotlib.pyplot as plt
# plt.scatter(df.Mileage,df[['Sell Price($)']])
# plt.scatter(df[['Age(yrs)']],df[['Sell Price($)']])
# plt.show()

x=df[['Mileage','Age(yrs)']]
y=df[['Sell Price($)']]

from sklearn.model_selection import train_test_split
x_train,x_test, y_train,y_test=train_test_split(x,y,test_size=0.2)

from sklearn.linear_model import LinearRegression
clf=LinearRegression()

clf.fit(x_train,y_train)

print(clf.predict(x_test))
print(y_test)
print(clf.score(x_test,y_test))