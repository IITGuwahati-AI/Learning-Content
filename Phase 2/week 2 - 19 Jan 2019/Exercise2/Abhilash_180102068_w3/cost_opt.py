#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 10:24:16 2019

@author: abhilash
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op
import utils    
data=np.loadtxt('./ex2data1.txt',delimiter=",");
x=data[:,:2]
y=data[:,2]
#y=y.reshape(np.size(y),1)
x=np.concatenate((np.array([1.0 for _ in range(y.shape[0])])[:,np.newaxis],x),axis=1)
m,n=x.shape
init_theta=np.zeros((n),dtype=float)

def plotData(x,y):
	fig=plt.figure()
	pos = y == 1
	neg = y == 0
	plt.plot(x[pos, 0], x[pos, 1], 'k*', lw=2, ms=10)
	plt.plot(x[neg, 0], x[neg, 1], 'ko', mfc='y', ms=8, mec='k', mew=1)
	plt.xlabel('Exam 1 score')
	plt.ylabel('Exam 2 score')
	plt.legend(['admitted','Not admitted'])

def sigmoid(x):
	return 1/(1+np.exp(-x))
'''def hTheta(init_theta,x):
	return sigmoid(np.matmul(init_theta.T,x))'''
def cost_fxn(init_theta,x,y):
    m=y.size
    J=0
    grad=np.zeros(init_theta.shape)
    for i in range(m):
        J-=((y[i]*np.log(sigmoid(np.matmul(init_theta.T,x[i]))))+(1-y[i])*np.log(1-sigmoid(np.matmul(init_theta.T,x[i]))))
        grad+=((sigmoid(np.matmul(init_theta.T,x[i]))-y[i])*x[i])
    #grad = (1/m)*(np.matmul((hTheta(theta,X)-y),X))
    return J/m,grad/m
J,grad=cost_fxn(init_theta,x,y)

print('Cost at initial theta (zeros):',J)
print('Expected cost (approx): 0.693\n')
print('Gradient at initial theta (zeros):')
print(grad)
print('Expected gradients (approx):\n\t[-0.1000, -12.0092, -11.2628]\n')
'''
# Compute and display cost and gradient with non-zero theta
test_theta = np.array([[-24,0.2,0.2]])
test_theta = np.reshape(test_theta,(3,1))
test_cost,test_grad = cost_fxn(test_theta,x,y)
print('Cost at test theta:',test_cost)
print('Expected cost (approx): 0.218\n')
print('Gradient at test theta:')
print(test_grad)
print('Expected gradients (approx):\n\t[0.043, 2.566, 2.647]')
'''
#Optimizing 

def opt_min(init_theta,x,y):
    options= {'maxiter': 400}
    res =  op.minimize(cost_fxn,init_theta,(x,y),method='TNC',jac = True ,options=options)
    cost = res.fun
    theta= res.x
    return cost,theta
opt_cost,opt_theta = opt_min(init_theta,x,y)
print('optimized theta:',opt_theta)
print('Expected theta (approx):\n\t[-25.161, 0.206, 0.201]')
print('optimized theta:',opt_cost)
print('Expected cost (approx): 0.203\n')

utils.plotDecisionBoundary(plotData,init_theta,x,y)
plt.show()
def predict(init_theta,x):
	m=x.shape[0]
	p=np.zeros(m)
	for i in range(m):
		if np.dot(x[i], init_theta)>=0.5:
			p[i]=1
	return p
p=predict(opt_theta,x)
print('Train Accuracy: {:.2f} %'.format(np.mean(p == y) * 100))
print('Expected accuracy (approx): 89.00 %') 