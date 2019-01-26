import numpy as np
import pandas as pd

import matplotlib.pyplot as plt 
 


data=pd.read_csv('Data/ex1data1.txt')
X=data.iloc[:,0].values
y=data.iloc[:,1].values 

X=np.c_[  np.ones(  96  )  , X]    
theta1=[0.8 ,  0.1]
def cost(theta):
    a =X*theta
    h=np.sum(a,axis=1) 
    j=0
    for i in range(0,np.size(h,axis=0)):
        j=j+(1/2)*(h[i]-y[i])*(h[i]-y[i] ) 
    j=j/(np.size(h,axis=0))
    return j 



def gradient(num,alpha,theta):
    for i in range(0,num):
        a =X*theta
        h=np.sum(a,axis=1)-y
        h=h*alpha
        h=h/np.size(h)
        h= np.transpose(X)*h 
        theta=theta- np.sum(h,axis= 1 ) 
        print(cost(theta))
    
    return theta


optimum = gradient(50000,0.005,theta1 )

ypred=np.sum(X*optimum,axis= 1 )

plt.scatter(X[:,1],y)
plt.plot(X[:,1],ypred )