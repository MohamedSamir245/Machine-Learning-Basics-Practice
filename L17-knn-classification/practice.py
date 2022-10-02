from turtle import color
import pandas as pd
from sklearn.datasets import load_iris
iris=load_iris()

df=pd.DataFrame(iris.data,columns=iris.feature_names)
df['target']=iris.target
# print(df.head())

df0=df[:50]
df1=df[50:100]
df2=df[100:150]

import matplotlib.pyplot as plt
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.scatter(df0['sepal length (cm)'],df0['sepal width (cm)'],color="green",marker='+')
plt.scatter(df1['sepal length (cm)'],df1['sepal width (cm)'],color="blue",marker='*')
# plt.show()
from sklearn.model_selection import train_test_split 
X=df.drop(['target'],axis='columns')
y=df.target
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=1)

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train,y_train)
# print(knn.score(X_test,y_test))

from sklearn.metrics import confusion_matrix
y_pred=knn.predict(X_test)
cm=confusion_matrix(y_test,y_pred)
print(cm)

import seaborn as sn
plt.figure(figsize=(7,5))
sn.heatmap(cm,annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
# plt.show()
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))
