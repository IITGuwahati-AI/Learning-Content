#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 10:33:30 2019

@author: abhilash
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op
import utils

data=np.loadtxt('./ex2data2.txt',delimiter=',')
x=data[:,:2]
y=data[:,2]

def plotData(x,y):
	fig=plt.figure()
	pos = y == 1
	neg = y == 0
	plt.plot(x[pos, 0], x[pos, 1], 'k*', lw=2, ms=10)
	plt.plot(x[neg, 0], x[neg, 1], 'ko', mfc='y', ms=8, mec='k', mew=1)
	plt.xlabel('Exam 1 score')
	plt.ylabel('Exam 2 score')
	plt.legend(['admitted','Not admitted'])
#visualize the data
plotData(x, y)
# Labels and Legend
plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')

# Specified in plot order
plt.legend(['y = 1', 'y = 0'], loc='upper right')
#pyplot.savefig('reg-data-plot')
plt.show()
#x=np.concatenate((np.array([1.0 for _ in range(y.shape[0])])[:,np.newaxis],x),axis=1)
m,n=x.shape
    
x = utils.mapFeature(x[:,0],x[:,1])
init_theta=np.zeros(x.shape[1],dtype=float)
#init_theta= np.reshape(init_theta,(x.shape[1],1 ))

def sigmoid(x):
	return 1/(1+np.exp(-x))
'''def hTheta(init_theta,x):
	return sigmoid(np.matmul(init_theta.T,x))'''

def costfxn_reg(init_theta,x,y,lambda_):
    m,n=x.shape
    J1=0;J2=0
    grad=np.zeros(init_theta.shape)
    for i in range(m):
        J1-=((y[i]*np.log(sigmoid(np.matmul(init_theta.T,x[i]))))+(1-y[i])*np.log(1-sigmoid(np.matmul(init_theta.T,x[i]))))
        grad+=((sigmoid(np.matmul(init_theta.T,x[i]))-y[i])*x[i])
    grad = grad/m
    for i in range(1,n):
        J2+=init_theta[i]**2
    J2= J2*(lambda_/(2*m))
    grad[1:]+=(lambda_/m)*init_theta[1:]
    return (J1/m)+J2,grad
J, grad=costfxn_reg(init_theta,x,y,1)
print('Cost at initial theta (zeros):',J)
print('Expected cost (approx)       : 0.693\n')
print('Gradient at initial theta (zeros) - first five values only:')
print(grad[:5,])
print('Expected gradients (approx) - first five values only:')
print('\t[0.0085, 0.0188, 0.0001, 0.0503, 0.0115]\n')

# Compute and display cost and gradient
# with all-ones theta and lambda = 10
test_theta = np.ones(x.shape[1],dtype=float)
#test_theta = np.reshape(test_theta,(x.shape[1],1))
test_cost, test_grad = costfxn_reg(test_theta, x, y, 10)
print('------------------------------\n')
print('Cost at test theta    :',test_cost)
print('Expected cost (approx): 3.16\n')
print('Gradient at initial theta (zeros) - first five values only:')
print(test_grad[:5,])
print('Expected gradients (approx) - first five values only:')
print('\t[0.3460, 0.1614, 0.1948, 0.2269, 0.0922]')
print('------------------------------\n')


def predict(opt_theta,x):
    m=x.shape[0]
    p=np.zeros(m)
    for i in range(m):
        if np.dot(x[i], opt_theta)>=0.5:
            p[i]=1
            return p

for lambda_ in range(11):
    # set options for optimize.minimize
    options= {'maxiter': 100}
    res =  op.minimize(costfxn_reg,init_theta,(x,y,lambda_),method='TNC',jac=True,options=options)
    # the value of costFunction at optimized theta
    opt_cost = res.fun
    # the optimized theta is in the x property of the result
    opt_theta = res.x
    
    utils.plotDecisionBoundary(plotData,opt_theta, x, y)
    plt.xlabel('Microchip Test 1')
    plt.ylabel('Microchip Test 2')
    plt.legend(['y = 1', 'y = 0'])
    plt.grid(False)
    plt.title('lambda = %0.2f' % lambda_)

    # Compute accuracy on our training set
    p = predict(opt_theta, x)
    print('Train Accuracy (with lambda ='+str(lambda_)+'): %.1f %%' % (np.mean(p == y) * 100))