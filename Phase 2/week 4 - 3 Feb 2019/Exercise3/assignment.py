# used for manipulating directory paths
import os

# Scientific and vector computation for python
import numpy as np

# Plotting library
from matplotlib import pyplot

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
data = loadmat(os.path.join('Data', 'ex3data1.mat'))
X, y = data['X'], data['y'].ravel()

# set the zero digit to 0, rather than its mapped 10 in this dataset
# This is an artifact due to the fact that this dataset was used in 
# MATLAB where there is no index 0
y[y == 10] = 0

m = y.size

#Randomly select 100 data points to display
rand_indices = np.random.choice(m, 100, replace=False)
sel = X[rand_indices, :]

utils.displayData(sel)

def lrCostFunction(theta,X,y,lambda_):
	#Initialize some useful values
    m = y.size
    
    # convert labels to ints if their type is bool
    if y.dtype == bool:
        y = y.astype(int)
    
    # You need to return the following variables correctly
    J = 0
    grad = np.zeros(theta.shape)
    
    some=utils.sigmoid(np.dot(X,theta))
    g=lambda i:((sum((some-y)*X[:,i]))+lambda_*theta[i])/m
    grad = np.array([g(i) for i in range(theta.shape[0]) ])
    grad[0]-=lambda_*theta[0]/m

    some = -(y*np.log(some)+(1-y)*np.log((1-some)))
    J= (sum(some)+sum(theta[1:]**2)*(lambda_/2))/m
    
    return J,grad

#test values for the parameters theta
theta_t = np.array([-2, -1, 1, 2], dtype=float)

# test values for the inputs
X_t = np.concatenate([np.ones((5, 1)), np.arange(1, 16).reshape(5, 3, order='F')/10.0], axis=1)
# test values for the labels
y_t = np.array([1, 0, 1, 0, 1])

# test value for the regularization parameter
lambda_t = 3



J,grad = lrCostFunction(theta_t, X_t, y_t, lambda_t)
print('Cost         : {:.6f}'.format(J))
print('Expected cost: 2.534819')
print('-----------------------')
print('Gradients:')
print(' [{:.6f}, {:.6f}, {:.6f}, {:.6f}]'.format(*grad))
print('Expected gradients:')
print(' [0.146561, -0.548558, 0.724722, 1.398003]');

def oneVsAll(X, y, num_labels, lambda_):
	# Some useful variables
    m, n = X.shape
    
    # You need to return the following variables correctly 
    all_theta = np.zeros((num_labels, n +1))

    # Add ones to the X data matrix
    X = np.concatenate([np.ones((m, 1)), X], axis=1)

    for i in range(num_labels):
    	initial_theta=np.zeros(n+1)
    	# Set options for minimize
    	options = {'maxiter': 50}
    	# Run minimize to obtain the optimal theta. This function will
    	# return a class object where theta is in `res.x` and cost in `res.fun`
    	res = optimize.minimize(lrCostFunction,initial_theta,(X,y==i,lambda_),jac=True,method='TNC',options=options)
    	all_theta[i]=res.x
    	print(res.x)
    return all_theta

lambda_ = 0.1
all_theta = oneVsAll(X, y, num_labels, lambda_)
print(all_theta)
def predictOneVsAll(all_theta,X):
	m = X.shape[0];
	num_labels = all_theta.shape[0]
	# You need to return the following variables correctly
	p = np.zeros(m)
	# Add ones to the X data matrix
	X = np.concatenate([np.ones((m, 1)), X], axis=1)
	for i in range(m):
		a=[np.dot(X[i],all_theta[j]) for j in range(num_labels)]
		p[i]=np.argmax(a)
	return p
pred=predictOneVsAll(all_theta,X)
print('Training Set Accuracy: {:.2f}%'.format(np.mean(pred == y) * 100))

    


