import os
import numpy as np
from scipy import optimize
from scipy.io import loadmat
import utils

#======================================= Loading Data ================================
input_layer_size  = 400
num_labels = 10
data = loadmat(os.path.join('Data', 'ex3data1.mat'))
X, y = data['X'], data['y'].ravel()
y[y == 10] = 0
y=np.reshape(y,(5000,1))
m = y.size
#====================================== Visualising Data =============================
rand_indices = np.random.choice(m, 100, replace=False)
sel = X[rand_indices, :]
utils.displayData(sel)
#=================================== Testing Cost Function ===========================
theta_t = np.array([-2, -1, 1, 2], dtype=float)
X_t = np.concatenate([np.ones((5, 1)), np.arange(1, 16).reshape(5, 3, order='F')/10.0], axis=1)
y_t = np.array([1, 0, 1, 0, 1])
lambda_t = 3
#================================== Cost Function ====================================
def lrCostFunction(theta, X, y, lambda_):
    m = y.size
    if y.dtype == bool:
        y = y.astype(int)
    J = 0

    n=np.shape(theta)[0]
    theta=np.reshape(theta,(n,1))
    grad = np.zeros(theta.shape)
    y=np.reshape(y,(np.shape(y)[0],1))
    J=-(1/m)*(np.transpose(y).dot(np.log(utils.sigmoid(X.dot(theta))))+(np.transpose(1-y).dot(np.log(1-utils.sigmoid(X.dot(theta))))))+(lambda_/(2*m))*(np.transpose(theta[1:n]).dot(theta[1:n]))
    for i in range(0, n):
        a = 0
        for j in range(0, m):
            a = a + ((utils.sigmoid(np.dot(X[j, :], np.reshape(theta, (n, 1)))) - y[j]) * X[j, i])
        if i == 0:
            grad[i] = a / m
        else:
            grad[i] = a / m + (lambda_ / m) * theta[i]

    grad = np.reshape(grad,(1,n))
    return J, grad

J, grad = lrCostFunction(theta_t, X_t, y_t, lambda_t)
#============================ Testing Cost Fucntion ==========================
print('Cost         : ',J)
print('Expected cost: 2.534819')
print('-----------------------')
print('Gradients:')
print(grad)
print('Expected gradients:')
print(' [0.146561, -0.548558, 0.724722, 1.398003]')
#=========================== One vs All Classification =======================
def oneVsAll(X, y, num_labels, lambda_):
    m, n = X.shape
    all_theta = np.zeros((num_labels, n + 1))
    X = np.concatenate([np.ones((m, 1)), X], axis=1)
    for i in range(0,num_labels):
        initial_theta = np.zeros(n + 1)
        options = {'maxiter': 50}
        res = optimize.minimize(lrCostFunction,initial_theta,(X, (y == i), lambda_),jac=True,method='TNC',options=options)
        all_theta[i]=res.x
    return all_theta

lambda_ = 0.1
all_theta = oneVsAll(X, y, num_labels, lambda_)

#======================================== Predict =================================
def predictOneVsAll(all_theta, X):
    m = X.shape[0]
    num_labels = all_theta.shape[0]
    p = np.zeros(m)
    X = np.concatenate([np.ones((m, 1)), X], axis=1)
    for i in range(0,m):
        a=(all_theta,np.transpose(X[i,:]))
        p[i]=a.index(max(a))+1
    return p


pred = predictOneVsAll(all_theta, X)
print('Training Set Accuracy: ',np.mean(pred == y) * 100)