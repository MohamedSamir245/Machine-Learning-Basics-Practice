import pandas as pd
from sklearn.datasets import load_digits
digits=load_digits()

# print(digits.data[0].reshape(8,8))
from matplotlib import pyplot as plt

# plt.gray()
# plt.matshow(digits.data[0].reshape(8,8))
# plt.show()

import numpy as np
# print(np.unique(digits.target))

df=pd.DataFrame(digits.data,columns=digits.feature_names)

# print(df.describe())

x=df
y=digits.target

from sklearn.preprocessing import StandardScaler
scalar=StandardScaler()
x_scaled=scalar.fit_transform(x)

from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(x_scaled,y,test_size=0.2,random_state=30)

from sklearn.linear_model import LogisticRegression
model=LogisticRegression(max_iter=1000)
# model.fit(X_train,y_train)
# print(model.score(X_test,y_test))

from sklearn.decomposition import PCA
# pca=PCA(0.95)
# x_pca=pca.fit_transform(x)
# print(x_pca.shape)

########
pca=PCA(n_components=2)
x_pca=pca.fit_transform(x)


# print(x_pca)
# print(pca.explained_variance_ratio_)
# print(pca.n_components_)

X_train, X_test, y_train, y_test = train_test_split(x_pca,y,test_size=0.2,random_state=30)

model.fit(X_train,y_train)
print(model.score(X_test,y_test))


