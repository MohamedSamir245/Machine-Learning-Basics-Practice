import pandas as pd
from sklearn.datasets import load_wine
wine=load_wine()
# print(dir(wine))

df=pd.DataFrame(wine.data,columns=wine.feature_names)
df['target']=wine.target
# print(df.columns)
# print(df)

inputs1=df.drop('target',axis='columns')
target1=df.target


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(inputs1,target1,train_size=0.8)

from sklearn.naive_bayes import GaussianNB
model1=GaussianNB()
model1.fit(X_train,y_train)
print(model1.score(X_test,y_test))

from sklearn.naive_bayes import MultinomialNB
model2=MultinomialNB()
model2.fit(X_train,y_train)
print(model2.score(X_test,y_test))