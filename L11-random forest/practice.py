from dataclasses import dataclass
import pandas as pd
from sklearn.datasets import load_digits
digits=load_digits()

import matplotlib.pyplot as plt
plt.gray()
# for i in range(4):
#     plt.matshow(digits.images[i])
#     plt.show()

df=pd.DataFrame(digits.data)

#print(digits.data)

#print(digits.target)

df['target']=digits.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.drop(['target'],axis='columns'),df.target,train_size=0.8)

from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(n_estimators=40)
model.fit(X_train,y_train)

#print(model.score(X_test,y_test))

y_predicted=model.predict(X_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_predicted)
# print(cm)

import matplotlib.pyplot as plt
import seaborn as sn
plt.figure(figsize=(10,7))
sn.heatmap(cm,annot=True)
plt.xlabel('predicted')
plt.ylabel('Truth')
plt.show()

