# used for manipulating directory paths
import os

# Scientific and vector computation for python
import numpy as np

# Plotting library
from matplotlib import pyplot

# Optimization module in scipy
from scipy import optimize
import utils

# will be used to load MATLAB mat datafile format
from scipy.io import loadmat

# 20x20 Input Images of Digits
input_layer_size  = 400

# 10 labels, from 1 to 10 (note that we have mapped "0" to label 10)
num_labels = 10

#  training data stored in arrays X, y
data = loadmat(os.path.join('Data', 'ex3data1.mat'))
X, y = data['X'], data['y'].ravel()

# set the zero digit to 0, rather than its mapped 10 in this dataset
# This is an artifact due to the fact that this dataset was used in 
# MATLAB where there is no index 0
y[y == 10] = 0

m = y.size
rand_indices = np.random.choice(m, 100, replace=False)
sel = X[rand_indices, :]

utils.displayData(sel)
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
    if y.dtype == bool:
        y = y.astype(int)
    J = 0
    grad = np.zeros(theta.shape)
    k=np.dot(X,theta)
    k=utils.sigmoid(k)
    J=np.log(k)
    J=np.dot(J.T,y)
    p=np.log(1-k)
    p=np.dot(p.T,1-y)
    J=-J-p
    J=J/m
    w=np.dot(theta.T,theta)
    w=w-theta[0]*theta[0]
    w=lambda_*w/2
    w=w/m
    J=J+w
    grad=k-y
    grad=np.dot(X.T,grad)
    grad=grad/m
    q=lambda_*theta
    q=q/m
    q[0]=0
    grad=grad+q
    return J,grad
J, grad = lrCostFunction(theta_t, X_t, y_t, lambda_t)

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
    X = np.concatenate([np.ones((m, 1)), X], axis=1)
    k=num_labels
    for i in range(0,k):
        initial_theta = np.zeros(n+1)
        options= {'maxiter': 400}
        res = optimize.minimize(lrCostFunction,initial_theta,(X,y==i,lambda_),jac=True,method='TNC',options=options)
        all_theta[i,:] =(res.x).T
    return all_theta

lambda_ = 0.1
all_theta = oneVsAll(X, y, num_labels, lambda_) 

def predictOneVsAll(all_theta, X):
    m = X.shape[0];
    num_labels = all_theta.shape[0]
    p = np.zeros(m)
    X = np.concatenate([np.ones((m, 1)), X], axis=1)
    p = np.argmax(X.dot(all_theta.T), axis=1)
    return p
pred = predictOneVsAll(all_theta, X)
print('Training Set Accuracy: {:.2f}%'.format(np.mean(pred == y) * 100))

data = loadmat(os.path.join('Data', 'ex3data1.mat'))
X, y = data['X'], data['y'].ravel()
y[y == 10] = 0
m = y.size
indices = np.random.permutation(m)
rand_indices = np.random.choice(m, 100, replace=False)
sel = X[rand_indices, :]

utils.displayData(sel)
input_layer_size  = 400  
hidden_layer_size = 25  
num_labels = 10 
weights = loadmat(os.path.join('Data', 'ex3weights.mat'))
Theta1, Theta2 = weights['Theta1'], weights['Theta2']
Theta2 = np.roll(Theta2, 1, axis=0)


def predict(Theta1, Theta2, X):
    if X.ndim == 1:
        X = X[None]
    m = X.shape[0]
    num_labels = Theta2.shape[0]
    p = np.zeros(X.shape[0])
    a1= np.concatenate([np.ones((m, 1)), X], axis=1)
    a2=utils.sigmoid(np.dot(a1,Theta1.T))
    a2=np.concatenate([np.ones((a2.shape[0],1)),a2],axis=1)
    a3=utils.sigmoid(np.dot(a2,Theta2.T))
    p=np.argmax(a3,axis=1)
    return p
    
pred = predict(Theta1, Theta2, X)
print('Training Set Accuracy: {:.1f}%'.format(np.mean(pred == y) * 100))

if indices.size > 0:
    i, indices = indices[0], indices[1:]
    utils.displayData(X[i, :], figsize=(4, 4))
    pred = predict(Theta1, Theta2, X[i, :])
    print('Neural Network Prediction: {}'.format(*pred))
else:
    print('No more images to display!')
pyplot.show()