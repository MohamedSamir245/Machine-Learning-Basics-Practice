import pandas as pd
from matplotlib import pyplot as plt
df=pd.read_csv("D:\Machine learning\ml_codebasics\L8-ligistic regression\one\ex\HR_comma_sep.csv")

left=df[df.left==1]
#print(left.shape)

retained=df[df.left==0]
#print(retained.shape)

#print(df.groupby('left').mean())

#pd.crosstab(df.salary,df.left).plot(kind='bar')
#pd.crosstab(df.Department,df.left).plot(kind='bar')

#plt.show()
subdf=df[['satisfaction_level','average_montly_hours','promotion_last_5years','salary']]

salary_dummies=pd.get_dummies(subdf.salary,prefix="salary")
df_with_dummies=pd.concat([subdf,salary_dummies],axis='columns')
#print(df_with_dummies.head)

df_with_dummies.drop('salary',axis='columns',inplace=True)
#print(df_with_dummies.head)

x=df_with_dummies
y=df.left

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x,y,train_size=0.8)

from sklearn.linear_model import LogisticRegression
model=LogisticRegression()

model.fit(X_train,y_train)

print(model.predict(X_test))
print(model.score(X_test,y_test))
