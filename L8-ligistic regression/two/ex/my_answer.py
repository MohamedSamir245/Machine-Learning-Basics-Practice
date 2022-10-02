import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris=load_iris()
#print(dir(iris))



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris.data,iris.target,train_size=0.8)

#print(len(X_train))
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(X_train,y_train)

#print(model.predict([iris.data[115]]))
#print(model.score(X_test,y_test))
# plt.matshow([iris.data[67]])
# plt.show()
#print(model.predict([iris.data[0]]))
#print(model.target[0])

y_predicted=model.predict(X_test)
from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test,y_predicted)
#print(cm )

import seaborn as sn
plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()

