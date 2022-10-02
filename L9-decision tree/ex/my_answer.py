from operator import truediv
import pandas as pd
df=pd.read_csv("D:\Machine learning\ml_codebasics\L9-decision tree\ex\extitanic.csv")

inputs=df.drop(['PassengerId','Name','Survived','SibSp','Parch','Ticket','Cabin','Embarked'],axis='columns')
target=df.Survived

from sklearn.preprocessing import LabelEncoder
le_sex=LabelEncoder()

inputs['Sex_n']=le_sex.fit_transform(inputs['Sex'])
inputs.drop(['Sex'],axis='columns',inplace=True)

#another solution
#inputs.Sex = inputs.Sex.map({'male': 1, 'female': 2})

inputs[['Age']]=inputs[['Age']].fillna(inputs[['Age']].mean())

#print(inputs)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(inputs,target,train_size=0.8)

from sklearn import tree
model=tree.DecisionTreeClassifier()
model.fit(X_train,y_train)

#print(model.predict([[3,22,7.25,1]]))

print(model.score(X_test,y_test))