from turtle import color
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
df=pd.read_csv("D:\Machine learning\ml_codebasics\L13-kmeans\income.csv")

#print(df)

# plt.scatter(df.Age,df[['Income($)']])
plt.xlabel('Age')
plt.ylabel('income($)')
# plt.show()

scaler=MinMaxScaler()
scaler.fit(df[['Income($)']])
df[['Income($)']]=scaler.transform(df[['Income($)']])

scaler.fit(df[['Age']])
df[['Age']]=scaler.transform(df[['Age']])
# print(df)

km=KMeans(n_clusters=3)

y_predicted=km.fit_predict(df[['Age','Income($)']])
# print(y_predicted)

df['cluster']=y_predicted
#print(df)

# print(km.cluster_centers_)

# df1=df[df.cluster==0]
# df2=df[df.cluster==1]
# df3=df[df.cluster==2]
# plt.scatter(df1.Age,df1[['Income($)']],color='red')
# plt.scatter(df2.Age,df2[['Income($)']],color='blue')
# plt.scatter(df3.Age,df3[['Income($)']],color='yellow')
# plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',marker='+')
# plt.legend()
# plt.show()

k_rng=range(1,10)
sse=[]

for k in k_rng:
    km=KMeans(n_clusters=k)
    km.fit(df[['Age','Income($)']])
    sse.append(km.inertia_)

# print(sse)

plt.xlabel('k')
plt.ylabel('Sum of Square error')
plt.scatter(k_rng,sse)

plt.show()