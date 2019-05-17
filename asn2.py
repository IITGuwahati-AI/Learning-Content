# -*- coding: utf-8 -*-
"""
Created on Thu May 16 08:55:32 2019

@author: HP
"""

# -*- coding: utf-8 -*-
"""
Created on Wed May 15 13:14:49 2019

@author: HP
"""
#importing numpy
import numpy as np
#Importing pyplot 
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
data=np.loadtxt('ex1data1.txt',delimiter=',')
#print(data)
plt.scatter(data[:,0], data[:,1], color='g')
plt.title('Scatter Plot of Training Data')
plt.ylabel('Profit in 10,000$')
plt.xlabel('Population of The city in 10,000s')
plt.show()
plt.savefig('Scatter Plot of Training Data')
m=np.size(data,0)
n=np.size(data,1)
def computecost(theta,data):
        temp=0
        data2=np.insert(data,0,1,axis = 1)
        for i in range(0,m):
            temp+=((np.matmul(np.transpose(theta),np.transpose(data2[i,:2])))-data[i,1])*((np.matmul(np.transpose(theta),np.transpose(data2[i,:2])))-data[i,1])
        return temp/(2*m)

print(computecost([[0],[0]],data))

#Gradient Descent
model = LinearRegression(fit_intercept=True)
model.fit(data[:,0][:, np.newaxis], data[:,1])
xfity = np.linspace(0, 20, 1000)
yfity = model.predict(xfity[:, np.newaxis])
plt.scatter(data[:,0], data[:,1],color='red')
plt.plot(xfity, yfity)
plt.title('Prediction')
plt.ylabel('Profit in 10,000$')
plt.xlabel('Population of The city in 10,000s')
plt.show
plt.savefig('Prediction')

print("Model slope:    ", model.coef_[0])
print("Model intercept:", model.intercept_)
#prediction
print(model.predict([[3.5],[7]]))



 
    
    

    