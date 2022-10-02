from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV


import pandas as pd

from sklearn.datasets import load_digits
digits=load_digits()

df=pd.DataFrame(digits.data,columns=digits.feature_names)
df['num']=digits.target

# print(df.head())

from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(digits.data,digits.target,test_size=0.2)

model_params={
    'svm':{
        'model':svm.SVC(gamma='auto'),
        'params':{
            'C':[1,10,20],
            'kernel':['rbf','linear']
        }
    },
    'random_forest':{
        'model':RandomForestClassifier(),
        'params':{
            'n_estimators':[1,5,10]
        }
    },
    'logistic_regression':{
        'model':LogisticRegression(),
        'params':{
            'C':[1,5,10]
        }
    },
    'gaussian_nb':{
        'model':GaussianNB(),
        'params':{}
    },
    'multinomial_nb':{
        'model':MultinomialNB(),
        'params':{}
    },
    'decisiontree':{
        'model':DecisionTreeClassifier(),
        'params':{}

    }
}


scores=[]

for model_name,mp in model_params.items():
    clf=GridSearchCV(mp['model'],mp['params'],cv=5,return_train_score=False)
    clf.fit(digits.data,digits.target)
    scores.append({
        'model':model_name,
        'best_score':clf.best_score_,
        'best_params':clf.best_params_
    })

df=pd.DataFrame(scores,columns=['model','best_score','best_params'])
print(df)