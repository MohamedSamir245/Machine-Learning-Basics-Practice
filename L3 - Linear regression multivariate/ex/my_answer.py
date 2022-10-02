import pandas as pd
import numpy as np
from sklearn import linear_model
import math
from word2number import w2n

df=pd.read_csv("D:\Machine learning\ml_codebasics\L3 - Linear regression multivariate\ex\hiring.csv")

df.experience=df.experience.fillna('zero')

median_score=math.floor(df[['test_score(out of 10)']].mean())

df[['test_score(out of 10)']]=df[['test_score(out of 10)']].fillna(median_score)

df.experience=df.experience.apply(w2n.word_to_num)

reg=linear_model.LinearRegression()

reg.fit(df[['experience','test_score(out of 10)','interview_score(out of 10)']],df[['salary($)']])

print(reg.predict([[12,10,10]]))
