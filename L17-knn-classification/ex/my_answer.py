import pandas as pd
from sklearn.datasets import load_digits
digits=load_digits()

df=pd.DataFrame(digits.data,columns=digits.feature_names)
# print(df.head())

df['target']=digits.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.drop('target',axis=1),df.target,test_size=0.2,random_state=1)

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train,y_train)
print(knn.score(X_test,y_test))

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
y_pred=knn.predict(X_test)
cm=confusion_matrix(y_test,y_pred)
# print(cm)

import seaborn as sn
plt.figure(figsize=(7,5))
sn.heatmap(cm,annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
# plt.show()

from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))

