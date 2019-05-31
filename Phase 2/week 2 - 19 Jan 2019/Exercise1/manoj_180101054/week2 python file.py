# -*- coding: utf-8 -*-
"""
Created on Fri May 31 12:26:52 2019

@author: Manoj
"""

import os
import numpy as np
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
def warmUpExercise():
    A=np.eye(5)
    return A
warmUpExercise()
data=np.loadtxt('../Data/ex1data1.txt',delimiter=',')
X,y=data[:,0],data[:,1]
m=y.size
def plotData(x,y):
    pyplot.plot(x,y,'ro',ms=10,mec='k')
    pyplot.ylabel('profit in $10,000')
    pyplot.xlabel('population of city in 10,000s')
plotData(X,y)
X = np.stack([np.ones(m), X], axis=1)
def computeCost(X,y,theta):
    m=y.size
    J=0
    J=np.dot(X,theta)-y
    J=np.dot(J,J)
    J=J/2
    J=J/m
    return J
J=computeCost(X,y,theta=np.array([0.0,0.0]))
print('With theta = [0,0]\nCost computed=%.2f'%J)
print('Expected cost value 32.07\n')
J=computeCost(X,y,theta=np.array([-1,2]))
print('With theta = [-1,2]\nCost computed=%.2f'%J)
print('Expected cost value 52.24\n')
def gradientDescent(X,y,theta,alpha,num_iters):
    m=y.shape[0]
    theta=theta.copy()
    J_history=[]
    for i in range(num_iters):
        theta=theta.copy()
        theta=theta.copy()-alpha*np.dot(X.T,np.dot(X,theta)-y)/m
        J_history.append(computeCost(X,y,theta))
    return theta,J_history    
theta=np.zeros(2)
iterations=1500
alpha=0.01
theta,J_history=gradientDescent(X,y,theta,alpha,iterations)
print('Theta found by gradient descent: {:.4f}, {:.4f}'.format(*theta))
print('Expected theta values (approximately): [-3.6303, 1.1664]')
plotData(X[:, 1], y)
pyplot.plot(X[:, 1], np.dot(X, theta), '-')
pyplot.legend(['Training data', 'Linear regression']);
predict1 = np.dot([1, 3.5], theta)
print('For population = 35,000, we predict a profit of {:.2f}\n'.format(predict1*10000))
predict2 = np.dot([1, 7], theta)
print('For population = 70,000, we predict a profit of {:.2f}\n'.format(predict2*10000))
theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)
J_vals = np.zeros((theta0_vals.shape[0], theta1_vals.shape[0]))
for i, theta0 in enumerate(theta0_vals):
    for j, theta1 in enumerate(theta1_vals):
        J_vals[i, j] = computeCost(X, y, [theta0, theta1])
J_vals = J_vals.T        
fig = pyplot.figure(figsize=(12, 5))
ax = fig.add_subplot(121, projection='3d')
ax.plot_surface(theta0_vals, theta1_vals, J_vals, cmap='viridis')
pyplot.xlabel('theta0')
pyplot.ylabel('theta1')
pyplot.title('Surface')
ax = pyplot.subplot(122)
pyplot.contour(theta0_vals, theta1_vals, J_vals, linewidths=2, cmap='viridis', levels=np.logspace(-2, 3, 20))
pyplot.xlabel('theta0')
pyplot.ylabel('theta1')
pyplot.plot(theta[0], theta[1], 'ro', ms=10, lw=2)
pyplot.title('Contour, showing minimum')
pass
data = np.loadtxt(, delimiter=',')
X = data[:, :2]
y = data[:, 2]
m = y.size
print('{:>8s}{:>8s}{:>10s}'.format('X[:,0]', 'X[:, 1]', 'y'))
print('-'*26)
for i in range(10):
    print('{:8.0f}{:8.0f}{:10.0f}'.format(X[i, 0], X[i, 1], y[i]))
def  featureNormalize(X):
    X_norm = X.copy()
    mu = np.zeros(X.shape[1])
    sigma = np.zeros(X.shape[1])
    m1=np.mean(X_norm,axis=0)
    mu=mu+m1
    s1=np.std(X_norm,axis=0)
    sigma=s1+sigma
    for k in range(0,X.shape[1]):
        X_norm[:,k]= (X[:,k]-np.ones(X.shape[0])*mu[k])/sigma[k]
    
    return X_norm,mu,sigma  
X_norm, mu, sigma = featureNormalize(X)
print('Computed mean:', mu)
print('Computed standard deviation:', sigma)
X = np.concatenate([np.ones((m, 1)), X_norm], axis=1)
def computeCostMulti(X, y, theta):
    m = y.shape[0]
    J=0
    J=np.dot(X,theta)-y
    J=np.dot(J,J)
    J=J/2
    J=J/m
    return J
def gradientDescentMulti(X, y, theta, alpha, num_iters):
    
    m = y.shape[0]
    theta = theta.copy()
    J_history = []
    for i in range(num_iters):
        theta=theta.copy()
        theta=theta.copy()-alpha*np.dot(X.T,np.dot(X,theta)-y)/m
        J_history.append(computeCost(X,y,theta))
    return theta,J_history
alpha = 0.1
num_iters = 400

# init theta and run gradient descent
theta = np.zeros(3)
theta, J_history = gradientDescentMulti(X, y, theta, alpha, num_iters)

# Plot the convergence graph
pyplot.plot(np.arange(len(J_history)), J_history, lw=2)
pyplot.xlabel('Number of iterations')
pyplot.ylabel('Cost J')

# Display the gradient descent's result
print('theta computed from gradient descent: {:s}'.format(str(theta)))
price = 0
print('Predicted price of a 1650 sq-ft, 3 br house (using gradient descent): ${:.0f}'.format(price))
data = np.loadtxt('../Data/ex1data2.txt', delimiter=',')
X = data[:, :2]
y = data[:, 2]
m = y.size
X = np.concatenate([np.ones((m, 1)), X], axis=1)
def normalEqn(X, y):
    theta = np.zeros(X.shape[1])
    theta1=np.dot(X.T,X)
    theta1=np.linalg.pinv(theta1)
    theta1=np.dot(theta1,X.T)
    theta=np.dot(theta1,y)
    return theta
theta = normalEqn(X, y);
print('Theta computed from the normal equations: {:s}'.format(str(theta)));
price = np.dot([1,1650,3],theta)
print('Predicted price of a 1650 sq-ft, 3 br house (using normal equations): ${:.0f}'.format(price))
