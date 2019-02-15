#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 20:52:32 2019

@author: abhilash
"""
#import os
import numpy as np
import matplotlib as plt
# Optimization module in scipy
from scipy import optimize

# will be used to load MATLAB mat datafile format
from scipy.io import loadmat

# library written for this exercise providing additional functions for assignment submission, and others
import utils
# 20x20 Input Images of Digits
input_layer_size  = 400

# 10 labels, from 1 to 10 (note that we have mapped "0" to label 10)
num_labels = 10

#  training data stored in arrays X, y
data = loadmat('ex3data1.mat')
X, y = data['X'], data['y'].ravel()
# set the zero digit to 0, rather than its mapped 10 in this dataset
# This is an artifact due to the fact that this dataset was used in 
# MATLAB where there is no index 0
y[y == 10] = 0

m = y.size
# test values for the parameters theta
theta_t = np.array([-2, -1, 1, 2], dtype=float)

# test values for the inputs
X_t = np.concatenate([np.ones((5, 1)), np.arange(1, 16).reshape(5, 3, order='F')/10.0], axis=1)

# test values for the labels
y_t = np.array([1, 0, 1, 0, 1])

# test value for the regularization parameter
lambda_t = 3

def lrCostFunction(theta, X, y, lambda_):
    m = y.size
    # convert labels to ints if their type is bool
    if y.dtype == bool:
        y = y.astype(int)
    J = 0
    h = utils.sigmoid(np.matmul(X,theta))
    grad = np.zeros(theta.shape)
    grad = (1/m)*np.matmul(X.T,(h-y))+(lambda_/m)*theta
    grad[0] -= (lambda_/m)*theta[0]
    J = np.matmul(y, np.log(h/(1-h)))+sum(np.log(1-h))
    J /= -1*m
    J = J + (lambda_/(2*m))*(np.matmul(theta[1:], theta[1:]))
    return J, grad

J,grad = lrCostFunction(theta_t, X_t, y_t, lambda_t)
print('Cost         : {:.6f}'.format(J))
print('Expected cost: 2.534819')
print('-----------------------')
print('Gradients:')
print(' [{:.6f}, {:.6f}, {:.6f}, {:.6f}]'.format(*grad))
print('Expected gradients:')
print(' [0.146561, -0.548558, 0.724722, 1.398003]');

def oneVsAll(X, y, num_labels, lambda_):
    m, n = X.shape
    all_theta = np.zeros((num_labels, n + 1))
    initial_theta = np.zeros(n + 1)
    # Add ones to the X data matrix
    X = np.concatenate([np.ones((m, 1)), X], axis=1)
    for i in range(num_labels):
            options = {'maxiter': 50}
            res = optimize.minimize(lrCostFunction,initial_theta,(X,(y==i), lambda_),jac=True,method='TNC',options=options) 
            all_theta[i]= res.x
    return all_theta
all_theta = oneVsAll(X, y, num_labels, 0.1)

def predictOneVsAll(all_theta, X):
    m = X.shape[0];
    num_labels = all_theta.shape[0]
    p = np.zeros(m)
    # Add ones to the X data matrix
    X = np.concatenate([np.ones((m, 1)), X], axis=1)
    p = np.matmul(X, all_theta.T)
    p = utils.sigmoid(p)
    p = np.array([np.argmax(j) for j in p])
    return p
pred_ = predictOneVsAll(all_theta, X)
print('Training Set Accuracy: {:.2f}%'.format(np.mean(pred_ == y) * 100))
print('-----------------------')
#Neural Networks

hidden_layer_size = 25   # 25 hidden units
weights = loadmat('ex3weights.mat')

# get the model weights from the dictionary
# Theta1 has size 25 x 401
# Theta2 has size 10 x 26
Theta1, Theta2 = weights['Theta1'], weights['Theta2']

# swap first and last columns of Theta2, due to legacy from MATLAB indexing, 
# since the weight file ex3weights.mat was saved based on MATLAB indexing
Theta2 = np.roll(Theta2, 1, axis=0)

def predict(Theta1, Theta2, X):
    # Make sure the input has two dimensions
    if X.ndim == 1:
        X = X[None]  # promote to 2-dimensions
    
    # useful variables
    m = X.shape[0]
    num_labels = Theta2.shape[0]

    # You need to return the following variables correctly 
    p = np.zeros(X.shape[0])
    X = np.concatenate([np.ones((m, 1)), X], axis=1)
    a2 = utils.sigmoid(np.matmul(X,Theta1.T))
    a2 = np.concatenate([np.ones((m, 1)), a2], axis=1)
    p = utils.sigmoid(np.matmul(a2, Theta2.T))
    p = np.array([np.argmax(j) for j in p])
    return p
pred = predict(Theta1, Theta2, X)
print('Training Set Accuracy: {:.1f}%'.format(np.mean(pred == y) * 100))

indices = np.random.permutation(m)
if indices.size > 0:
    i, indices = indices[0], indices[1:]
    utils.displayData(X[i, :], figsize=(4, 4))
    pred = predict(Theta1, Theta2, X[i, :])
    print('Neural Network Prediction: {}'.format(*pred))
else:
    print('No more images to display!')