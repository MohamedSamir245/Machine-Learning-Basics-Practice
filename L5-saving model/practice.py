import numpy as np 
import pandas as pd
from sklearn import linear_model
import joblib
import pickle

df=pd.read_csv("D:\Machine learning\ml_codebasics\L5-saving model\homeprices.csv")

reg=linear_model.LinearRegression()
reg.fit(df[['area']],df.price)

# print(reg.predict([[5000]]))

# with open('D:\Machine learning\L5\model_pickle.pkl','wb') as file:
#     pickle.dump(reg,file)

# with open('.\model_pickle.pkl', 'rb') as f:
#     mp = pickle.load(f)
# print(mp.predict([[5000]]))

joblib.dump(reg,'D:\Machine learning\ml_codebasics\L5-saving model\model_joblib')

mj=joblib.load('D:\Machine learning\ml_codebasics\L5-saving model\model_joblib')
print(mj.predict([[5000]]))

