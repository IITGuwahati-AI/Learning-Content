#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 14:11:49 2019

@author: abhilash
"""

import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('ex1data2.txt',delimiter = ',')
X = data[:,:2]
y = data[:,2]
m = y.shape[0]
X = np.concatenate((X,np.ones((1,m)).T),axis = 1)
mu = np.mean(X)
sig = np.std(X)
#feature normalization
def feat_norm(X):
    mu = np.mean(X)
    sig = np.std(X)
    t = np.ones((X.shape[0],1))
    X_norm = (X - (t*mu))/(t*sig)
    return X_norm

#h = t0*x1+t1*x2+t2
#computing cost fxn
def comp_cost_mult(X,y,theta):
    m = y.shape[0]
    J = 0
    h = np.matmul(X,theta)
    for i in range (0,m):
        J += (1/(2*m))*((h[i,0]-y[i])**2)
    return J
#Gradient Descent
alpha = 0.01
num_iters = 1000
theta = np.zeros((3,1))
def grad_desc_mult(X,y,theta,alpha,num_iters):
    m = y.shape[0]
    J_his = np.zeros((num_iters,1))
    h = np.matmul(X,theta)
    for i in range(0,m):
       t0 = (h[i,0]-y[i])*X[i,0]
       t1 = (h[i,0]-y[i])*X[i,1]
       t2 = (h[i,0]-y[i])*X[i,2]
       theta[0] -= (alpha/m)*t0
       theta[1] -= (alpha/m)*t1
       theta[2] -= (alpha/m)*t2
       J_his[i,0] = comp_cost_mult(X,y,theta)
    return theta,J_his
theta,J_his = grad_desc_mult(X,y,theta,alpha,num_iters)
print(J_his.max())
print('Theta computed from gradient descent:'+str(theta))
k = np.linspace(1,1000,num = 1000)
plt.plot(k,J_his,'-b')
plt.xlabel('Number of iterations')
plt.ylabel('Cost J')
plt.savefig('iter_VS_costJ.png')
plt.show()

#Estimate the price of a 1650 sq-ft, 3 br house
d = np.array([1,1650,3])
d = (d - mu)/sig
price_g = np.matmul(d,theta)
print('Predicted price of a 1650 sq-ft, 3 br house(using gradient descent): $'+str(price_g[0]))

#normal equations
from numpy.linalg import inv
def norm_eqn(X,y):
    l = np.matmul(X.T,y)
    theta = np.matmul(inv(np.matmul(X.T,X)),l)
    return theta
price_n = np.matmul(d,norm_eqn(X,y))
print('Predicted price of a 1650 sq-ft, 3 br house(using normal euquations): $'+str(price_n))