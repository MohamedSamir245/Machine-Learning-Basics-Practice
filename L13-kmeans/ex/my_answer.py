import pandas as pd
from sklearn.datasets import load_iris
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

iris=load_iris()

df=pd.DataFrame(iris.data,columns=iris.feature_names)

df.drop(['sepal length (cm)','sepal width (cm)'],axis='columns',inplace=True)
# print(df)

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
scaler.fit(df[['petal length (cm)']])
df[['petal length (cm)']]=scaler.transform(df[['petal length (cm)']])

scaler.fit(df[['petal width (cm)']])
df[['petal width (cm)']]=scaler.transform(df[['petal width (cm)']])
# print(df)

km=KMeans(n_clusters=3)
yp=km.fit_predict(df[['petal length (cm)','petal width (cm)']])

df['cluster']=yp
print(df)

df1=df[df.cluster==0]
df2=df[df.cluster==1]
df3=df[df.cluster==2]

plt.scatter(df1['petal length (cm)'],df1['petal width (cm)'],color='red')
plt.scatter(df2['petal length (cm)'],df2['petal width (cm)'],color='blue')
plt.scatter(df3['petal length (cm)'],df3['petal width (cm)'],color='yellow')
plt.show()

# k_rng=range(1,10)
# sse=[]
# for k in k_rng:
#     km=KMeans(n_clusters=k)
#     km.fit(df[['petal length (cm)']],df[['petal width (cm)']])
#     sse.append(km.inertia_)

# plt.xlabel('k')
# plt.ylabel('Sum of square error')
# plt.scatter(k_rng,sse)
# plt.show()