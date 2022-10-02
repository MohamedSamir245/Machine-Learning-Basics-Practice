import pandas as pd
import numpy as np
import math
df=pd.read_csv("D:\Machine learning\ml_codebasics\L4 - gradient descent\ex\scores.csv")

def gradient_descent(x,y):
    m_curr=b_curr=0
    iterations=1000000
    n=len(x)
    learning_rate=0.0001
    pre_cost=0
    
    for i in range(iterations):
        y_predicted=m_curr*x+b_curr
        cost=(1/n)*sum([val**2 for val in (y-y_predicted)])
        md=-(2/n)*sum(x*(y-y_predicted))
        bd=-(2/n)*sum(y-y_predicted)
        m_curr=m_curr-md*learning_rate
        b_curr=b_curr-bd*learning_rate
        if(math.isclose(cost,pre_cost,rel_tol=1e-20)):
            break
        pre_cost=cost
        print("m {}, b {}, cost {}, iteration {}".format(m_curr,b_curr,cost,i))









x=np.array(df.math)
y=np.array(df.cs)

gradient_descent(x,y)

