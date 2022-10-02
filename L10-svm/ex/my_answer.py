import pandas as pd
from sklearn.datasets import load_digits
digits=load_digits()
#print(dir(digits))

df = pd.DataFrame(digits.data,digits.target)
df['target']=digits.target

x=df.drop(['target'],axis='columns')
y=df.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x,y,train_size=0.8)

# from sklearn.svm import SVC
# rbf_model=SVC(kernel='rbf')

# rbf_model.fit(X_train,y_train)
# print(rbf_model.score(X_test,y_test))

from sklearn.svm import SVC
ln_model=SVC(kernel='linear')

ln_model.fit(X_train,y_train)
print(ln_model.score(X_test,y_test))