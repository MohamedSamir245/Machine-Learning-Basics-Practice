from cgi import test
import pandas as pd
df=pd.read_csv("D:\Machine learning\ml_codebasics\L20-bagging\diabetes.csv")

#checking the data
# print(df.isnull().sum())
# print(df.describe())
# print(df.Outcome.value_counts())

x=df.drop('Outcome',axis=1)
y=df.Outcome

#scaling the data as we see it needs a scale from the describe
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_scaled=scaler.fit_transform(x)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled,y,stratify=y,random_state=10)

# print(y_train.value_counts())
# print(201/375)

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
# scores=cross_val_score(DecisionTreeClassifier(),x,y,cv=5)
# print(scores)
# print(scores.mean())

from sklearn.ensemble import BaggingClassifier
# bag_model=BaggingClassifier(base_estimator=DecisionTreeClassifier(),
# n_estimators=100,max_samples=0.8,oob_score=True,random_state=0
# )

# bag_model.fit(X_train,y_train)
# print(bag_model.oob_score_)
# print(bag_model.score(X_test,y_test))

bag_model=BaggingClassifier(base_estimator=DecisionTreeClassifier(),
n_estimators=100,max_samples=0.8,oob_score=True,random_state=0
)
# scores=cross_val_score(bag_model,x,y,cv=5)
# print(scores) 
# print(scores.mean())

from sklearn.ensemble import RandomForestClassifier
scores=cross_val_score(RandomForestClassifier(),x,y,cv=5)
print(scores) 
print(scores.mean())

