from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.datasets import load_digits
import pandas as pd

digits=load_digits()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =train_test_split(digits.data,digits.target,test_size=0.3)

# lr=LogisticRegression()
# lr.fit(X_train,y_train)
# print(lr.score(X_test,y_test))

# svm=SVC()
# svm.fit(X_train,y_train)
# print(svm.score(X_test,y_test))

# rf=RandomForestClassifier()
# rf.fit(X_train,y_train)
# print(rf.score(X_test,y_test))

from sklearn.model_selection import KFold
kf=KFold(n_splits=3)
# for train_index, test_index in kf.split([1,2,3,4,5,6,7,8,9]):
#     print(train_index,test_index)

def get_score(model,X_train,y_train,X_test,y_test):
    model.fit(X_train,y_train)
    return model.score(X_test,y_test)

# print(get_score(LogisticRegression(),X_train,y_train,X_test,y_test))
# print(get_score(SVC(),X_train,y_train,X_test,y_test))
# print(get_score(RandomForestClassifier(),X_train,y_train,X_test,y_test))

from sklearn.model_selection import StratifiedKFold
skf=StratifiedKFold(n_splits=3)

scores_l=[]
scores_svm=[]
scores_rf=[]

df=pd.DataFrame(digits.data)

for train_index, test_index in kf.split(df):
    X_train, X_test, y_train, y_test=digits.data[train_index],digits.data[test_index],digits.target[train_index],digits.target[test_index]
    scores_l.append(get_score(LogisticRegression(),X_train,y_train,X_test,y_test))
    scores_svm.append(get_score(SVC(),X_train,y_train,X_test,y_test))
    scores_rf.append(get_score(RandomForestClassifier(),X_train,y_train,X_test,y_test))

# print(scores_l)
# print(scores_svm)    
# print(scores_rf)

from sklearn.model_selection import cross_val_score
print(cross_val_score(LogisticRegression(),digits.data,digits.target))