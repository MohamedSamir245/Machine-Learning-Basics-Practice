import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv("D:\Machine learning\L8\one\insurance_data.csv")

# plt.scatter(df[['age']],df[['bought_insurance']],color='red',marker='+')
# plt.show()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df[['age']],df[['bought_insurance']],test_size=0.1)

from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(X_train,y_train)

print(X_test)
print(model.predict(X_test))
print(model.score(X_test,y_test))

print(model.predict_proba(X_test))