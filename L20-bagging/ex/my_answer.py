import pandas as pd
df=pd.read_csv("D:\Machine learning\ml_codebasics\L20-bagging\ex\heart.csv")
# print(df.isnull().sum())
# print(df[df.Cholesterol>(df.Cholesterol.mean()+3*df.Cholesterol.std())])
df1=df[df.Cholesterol<=(df.Cholesterol.mean()+3*df.Cholesterol.std())]
# print(df[df.MaxHR>(df.MaxHR.mean()+3*df.MaxHR.std())])

# print(df[df.FastingBS>(df.FastingBS.mean()+3*df.FastingBS.std())])

# print(df[df.Oldpeak>(df.Oldpeak.mean()+3*df.Oldpeak.std())])
df2=df[df.Oldpeak<=(df.Oldpeak.mean()+3*df.Oldpeak.std())]

# print(df[df.RestingBP>(df.RestingBP.mean()+3*df.RestingBP.std())])
df3 = df2[df2.RestingBP<=(df2.RestingBP.mean()+3*df2.RestingBP.std())]

# print(df.ChestPainType.unique())
# print(df.RestingECG.unique())
# print(df.ExerciseAngina.unique())
# print(df.ST_Slope.unique())

df4=df3.copy()
df4.ExerciseAngina.replace(
    {
        'N':0,
        'Y':1
    }, inplace=True
)
df4.ST_Slope.replace(
    {
        'Down': 1,
        'Flat': 2,
        'Up': 3
    },
    inplace=True
)
df4.RestingECG.replace(
    {
        'Normal': 1,
        'ST': 2,
        'LVH': 3
    },
    inplace=True)

df5=pd.get_dummies(df4,drop_first=True)

X = df5.drop("HeartDisease",axis='columns')
y = df5.HeartDisease

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=30)

from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

# scores=cross_val_score(SVC(),X,y,cv=5)
# print(scores.mean())

from sklearn.ensemble import BaggingClassifier

# bag_model = BaggingClassifier(base_estimator=SVC(), n_estimators=100, max_samples=0.8, random_state=0)
# scores = cross_val_score(bag_model, X, y, cv=5)
# print(scores.mean())

from sklearn.tree import DecisionTreeClassifier
# scores = cross_val_score(DecisionTreeClassifier(), X, y, cv=5)
# print(scores.mean())

bag_model = BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=100, max_samples=0.8, random_state=0)
scores = cross_val_score(bag_model, X, y, cv=5)
print(scores.mean())

