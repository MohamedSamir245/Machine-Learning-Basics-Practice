import pandas as pd
from sklearn.datasets import load_iris
iris=load_iris()

df=pd.DataFrame(iris.data,columns=iris.feature_names)
#print(df)

df['target']=iris.target
# print(df)

from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(df.drop(['target'],axis='columns'),df.target,train_size=0.8,random_state=10)

from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(n_estimators=10)
model.fit(X_train,y_train)

print(model.score(X_test,y_test))
