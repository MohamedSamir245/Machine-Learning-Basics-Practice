import pandas as pd
df=pd.read_csv("D:\Machine learning\ml_codebasics\L14-naive-bayes\one\mtitanic.csv")
df.drop(['Name','SibSp','Parch','Ticket','Cabin','Embarked','PassengerId'],axis='columns',inplace=True)
# print(df)

target=df.Survived
inputs=df.drop('Survived',axis='columns')

dummies=pd.get_dummies(inputs.Sex)
inputs=pd.concat([inputs,dummies],axis='columns')

inputs.drop('Sex',axis='columns',inplace=True)
# print(inputs.columns[inputs.isna().any()])
inputs[['Age']]=inputs[['Age']].fillna(inputs[['Age']].mean())
# print(inputs.columns[inputs.isna().any()])

# print(inputs)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(inputs,target,test_size=0.2)

from sklearn.naive_bayes import GaussianNB
model=GaussianNB()
model.fit(X_train,y_train)
# print(model.score(X_test,y_test))
