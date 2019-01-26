import numpy as np
import pandas as pd

import matplotlib.pyplot as plt 
 


my_data=pd.read_csv('Data/ex2data1.txt')
y=my_data.iloc[:, -1 ].values  
my_data = (my_data - my_data.mean())/my_data.std()
my_data.head()
X=my_data.iloc[:,:-1].values
   
X=np.c_[  np.ones( 99   )  , X]    
theta1=[0.5, 0.5,0.5]
def sigmoid(z):
    return (1/(1+np.exp(-1*z)))


def  cost(h, y):
    return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()



def gradient(num,alpha,theta):
    for i in range(0,num):
        a = sigmoid(np.sum(X*theta,axis=1)) 
        h=a -y
        h=h*alpha
        h=h/np.size(h)
        h= np.transpose(X)*h 
        theta=theta- np.sum(h,axis= 1 ) 
        print(cost(sigmoid(np.sum(X*theta,axis=1)),y ))
    return theta

optimum = gradient(100000   ,0.01 ,theta1) 

pred=   sigmoid(np.sum(X*optimum  ,axis=1)  ) 

for i in range(0,99 ):
    if pred[i] <0.5:
        pred[i]=0
    else :
        pred[i]=1


plt.scatter(X[:,1],X[:,2],c= y  )
plot_x = np.array([np.min(X[:, 1]) - 2, np.max(X[:, 1]) + 2])
plot_y = (-1. / optimum[2]) * (optimum [1] * plot_x + optimum[0])
plt.plot(plot_x, plot_y)


 