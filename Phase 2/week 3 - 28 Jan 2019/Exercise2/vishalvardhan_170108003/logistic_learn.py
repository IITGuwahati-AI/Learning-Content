import numpy as np
import matplotlib.pyplot as pyplot
from math import log
# Optimization module in scipy
from scipy import optimize
# library written for this exercise providing additional functions for assignment submission, and others
import utils
import os

def plotData(x,y):
    """
    Plots the data points X and y into a new figure. Plots the data 
    points with * for the positive examples and o for the negative examples.

    Parameters
    ----------
    X : array_like
    An Mx2 matrix representing the dataset. 

    y : array_like
    Label values for the dataset. A vector of size (M, ).

    Instructions
    ------------
    Plot the positive and negative examples on a 2D plot, using the
    option 'k*' for the positive examples and 'ko' for the negative examples.    
    """
    # Find Indices of Positive and Negative Examples
    pos = y == 1
    neg = y == 0

    pyplot.plot(x[pos, 0], x[pos, 1], 'k*', lw=2, ms=10)
    pyplot.plot(x[neg, 0], x[neg, 1], 'ko', mfc='y', ms=8, mec='k', mew=1)


def sigmoid(z):
    """
    Compute sigmoid function given the input z.

    Parameters
    ----------
    z : array_like
    The input to the sigmoid function. This can be a 1-D vector 
    or a 2-D matrix. 

    Returns
    -------
    g : array_like
    The computed sigmoid function. g has the same shape as z, since
    the sigmoid is computed element-wise on z.

    Instructions
    ------------
    Compute the sigmoid of each value of z (z can be a matrix, vector or scalar).
    """
    return 1/(1+np.exp(-z))

def costFunction(theta,X,y):
    """
    Compute cost and gradient for logistic regression. 

    Parameters
    ----------
    theta : array_like
    The parameters for logistic regression. This a vector
    of shape (n+1, ).

    X : array_like
    The input dataset of shape (m x n+1) where m is the total number
    of data points and n is the number of features. We assume the 
    intercept has already been added to the input.

    y : arra_like
    Labels for the input. This is a vector of shape (m, ).

    Returns
    -------
    J : float
    The computed value for the cost function. 

    grad : array_like
    A vector of shape (n+1, ) which is the gradient of the cost
    function with respect to theta, at the current values of theta.

    Instructions
    ------------
    Compute the cost of a particular choice of theta. You should set J to 
    the cost. Compute the partial derivatives and set grad to the partial
    derivatives of the cost w.r.t. each parameter in theta.
    """
    h = np.matmul(X, theta)
    h = sigmoid(h)
    m = len(y)
    f1 = lambda i:log(i/(1-i))
    f2 = lambda i:log(1-i)
    vf1 = np.vectorize(f1)
    vf2 = np.vectorize(f2)
    J = (-1/m)*(np.matmul(y,vf1(h)) + np.matmul(np.ones(m), vf2(h)))
    grad = (1/m)*np.matmul(h-y, X)
    return J,grad

def predict(theta, X):
    """
    Predict whether the label is 0 or 1 using learned logistic regression.
    Computes the predictions for X using a threshold at 0.5 
    (i.e., if sigmoid(theta.T*x) >= 0.5, predict 1)

    Parameters
    ----------
    theta : array_like
    Parameters for logistic regression. A vecotor of shape (n+1, ).

    X : array_like
    The data to use for computing predictions. The rows is the number 
    of points to compute predictions, and columns is the number of
    features.

    Returns
    -------
    p : array_like
    Predictions and 0 or 1 for each row in X. 

    Instructions
    ------------
    Complete the following code to make predictions using your learned 
    logistic regression parameters.You should set p to a vector of 0's and 1's    
    """
    m = X.shape[0] # Number of training examples

    # You need to return the following variables correctly
    p = sigmoid(np.matmul(X,theta))

    # ====================== YOUR CODE HERE ======================
    p = p + 0.5
    p = p.astype(int)
    # ============================================================
    return p

def costFunctionReg(theta, X, y, lambda_ = 1):
    """
    Compute cost and gradient for logistic regression with regularization.

    Parameters
    ----------
    theta : array_like
    Logistic regression parameters. A vector with shape (n, ). n is 
    the number of features including any intercept. If we have mapped
    our initial features into polynomial features, then n is the total 
    number of polynomial features. 

    X : array_like
    The data set with shape (m x n). m is the number of examples, and
    n is the number of features (after feature mapping).

    y : array_like
    The data labels. A vector with shape (m, ).

    lambda_ : float
    The regularization parameter. 

    Returns
    -------
    J : float
    The computed value for the regularized cost function. 

    grad : array_like
    A vector of shape (n, ) which is the gradient of the cost
    function with respect to theta, at the current values of theta.

    Instructions
    ------------
    Compute the cost `J` of a particular choice of theta.
    Compute the partial derivatives and set `grad` to the partial
    derivatives of the cost w.r.t. each parameter in theta.
    """
    # Initialize some useful values
    m = y.size  # number of training examples

    # You need to return the following variables correctly 

    # ===================== YOUR CODE HERE ======================
    J,grad = costFunction(theta,X,y)
    J += (lambda_/(2*m))*(np.matmul(theta,theta))
    J -= (lambda_/(2*m))*(theta[0]**2)
    grad += (lambda_/m)*theta
    grad[0] = grad[0] - (lambda_/m)*theta[0]
    # =============================================================
    return J, grad

