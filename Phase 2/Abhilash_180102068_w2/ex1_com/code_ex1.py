#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 17:57:10 2019

@author: abhilash
"""

import numpy as np
import matplotlib.pyplot as plt

print('printing elementary matrix')
A=np.identity(5)
print(A)

data = np.loadtxt('ex1data1.txt',delimiter = ',')

x = np.array([data[:,0]])
y = np.array([data[:,1]]).T
m = y.shape[0]
#Ploting
plt.plot(x.T,y,'r^')
plt.ylabel('Profit in $10,000s')
plt.xlabel('Population of City in 10,000s')
plt.title('Profit VS Population')
plt.savefig('Profit_VS_Population.png')
plt.show()

#Gradient Descent
X = np.concatenate((np.ones((m,1)),x.T),axis = 1)
theta = np.zeros((2,1))
num_iter = 1500
alpha = 0.01
def comp_cost(X,y,theta):
    J = 0
    h = X.dot(theta)
    for i in range(0,m):
        J += (1/2*m)*((h[i,0]-y[i])**2)
    return J;
#display intial cost
print(comp_cost(X,y,theta))
def grad_desc(X,y,theta,alpha,num_iter):
    J_his = np.zeros((num_iter,1))
    h = X.dot(theta)
    for i in range(0,m):
        t0 = h[i,0]-y[i]
        t1 = (h[i,0]-y[i])*X[i,1]
        theta[0] -= (alpha/m)*t0
        theta[1] -= (alpha/m)*t1
        J_his[i] = comp_cost(X,y,theta)
    #return J_his
    return theta
theta = grad_desc(X,y,theta,alpha,num_iter)
print('Theta found by gradient descent: Theta[0] = '+str(theta[0])+'; Theta[1]= '+str(theta[1]))
#ploting linear fit
plt.plot(x.T,y,'r^',label='Training data')
plt.plot(X[:,1],X.dot(theta),'-',label='Linear regerssion')
plt.ylabel('Profit in $10,000s')
plt.xlabel('Population of City in 10,000s')
plt.title('Profit VS Population')
plt.legend()
plt.savefig('linear_regr.png')
plt.show()
#Predict values for population sizes of 18,000 and 25,000
p1=np.array([1,1.8])
p2=np.array([1,2.5])
predict1 = p1.dot(theta)
predict2 = p2.dot(theta)
print('For population = 18,000, we predict a profit of '+str(predict1[0]*10000))
print('For population = 25,000, we predict a profit of '+str(predict2[0]*10000))

#Visualizing J(theta_0, theta_1)
theta0_vals = np.linspace(-10, 10,num = 100)
theta1_vals = np.linspace(-1, 4,num = 100)
J_vals = np.zeros((theta0_vals.shape[0],theta1_vals.shape[0]))
for i in range(0,theta0_vals.shape[0]):
    for j in range(0,theta1_vals.shape[0]):
        t = np.zeros((2,1))
        t[0,0] = theta0_vals[i]
        t[1,0] = theta1_vals[j]
        J_vals[i,j] = comp_cost(X, y, t)

J_vals = J_vals.T/10000
theta0_vals,theta1_vals = np.meshgrid(theta0_vals,theta1_vals)
#plotting surface plot
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
fig = plt.figure() 
ax = fig.add_subplot(111,projection='3d') 
ax.plot_surface(theta0_vals,theta1_vals,J_vals,cmap=cm.rainbow,linewidth=0, antialiased=False)
plt.xlabel('theta[0]')
plt.ylabel('theta[1]')
plt.savefig('surface_plot.png')
plt.show()
#plotting contour
plt.contour(theta0_vals,theta1_vals,J_vals,np.logspace(-2, 3, num=20))
plt.plot(theta[0],theta[1],'rx')
plt.xlabel('theta[0]')
plt.ylabel('theta[1]')
plt.savefig('contour.png')
plt.show()
