import os
import numpy as np
from matplotlib import pyplot as plt
import math
from scipy import optimize
import utils

data=np.genfromtxt("Data/ex2data2.txt",delimiter=',')
X,y =data[:,:2],data[:,2]
#X=np.concatenate((np.array([1.0 for _ in range(y.shape[0])])[:,np.newaxis],X),axis=1)
m,n=X.shape
def plotData(X,y):
	fig=plt.figure()
	pos = y == 1
	neg = y == 0
	plt.plot(X[pos, 0], X[pos, 1], 'k*', lw=2, ms=10)
	plt.plot(X[neg, 0], X[neg, 1], 'ko', mfc='y', ms=8, mec='k', mew=1)
	plt.xlabel('Microchip Test 1')
	plt.ylabel('Microchip Test 2')
	plt.legend(['y = 1', 'y = 0'], loc='upper right')
	

X = utils.mapFeature(X[:, 0], X[:, 1])

def sigmoid(z):
	return 1/(1+math.exp(-z))

def costFunctionReg(theta,X,y,lambda_):
	m,n=X.shape
	J1=0;J2=0
	grad=np.zeros(theta.shape)
	for i in range(m):
		J1-=((y[i]*np.log(sigmoid(np.matmul(theta.T,X[i]))))+(1-y[i])*np.log(1-sigmoid(np.matmul(theta.T,X[i]))))
		grad+=((sigmoid(np.matmul(theta.T,X[i]))-y[i])*X[i])
	J1/=m
	grad=grad/m
	grad[1:]+=(lambda_/m)*theta[1:]
	for i in range(1,n):
		J2+=theta[i]**2
	J2*=(lambda_/(2*m))
	return J1+J2,grad
# Initialize fitting parameters
initial_theta = np.zeros(X.shape[1])

# Set regularization parameter lambda to 1
# DO NOT use `lambda` as a variable name in python
# because it is a python keyword
lambda_ = 1

# Compute and display initial cost and gradient for regularized logistic
# regression
cost, grad = costFunctionReg(initial_theta, X, y, lambda_)

print('Cost at initial theta (zeros): {:.3f}'.format(cost))
print('Expected cost (approx)       : 0.693\n')

print('Gradient at initial theta (zeros) - first five values only:')
print('\t[{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}]'.format(*grad[:5]))
print('Expected gradients (approx) - first five values only:')
print('\t[0.0085, 0.0188, 0.0001, 0.0503, 0.0115]\n')


# Compute and display cost and gradient
# with all-ones theta and lambda = 10
test_theta = np.ones(X.shape[1])
cost, grad = costFunctionReg(test_theta, X, y, 10)


print('------------------------------\n')
print('Cost at test theta    : {:.2f}'.format(cost))
print('Expected cost (approx): 3.16\n')

print('Gradient at initial theta (zeros) - first five values only:')
print('\t[{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}]'.format(*grad[:5]))
print('Expected gradients (approx) - first five values only:')
print('\t[0.3460, 0.1614, 0.1948, 0.2269, 0.0922]')

# Initialize fitting parameters
initial_theta = np.zeros(X.shape[1])

# Set regularization parameter lambda to 1 (you should vary this)
lambda_ = 1

# set options for optimize.minimize
options= {'maxiter': 100}

res = optimize.minimize(costFunctionReg,
                        initial_theta,
                        (X, y, lambda_),
                        jac=True,
                        method='TNC',
                        options=options)

# the fun property of OptimizeResult object returns
# the value of costFunction at optimized theta
cost = res.fun

# the optimized theta is in the x property of the result
theta = res.x

def predict(theta,X):
	m=X.shape[0]
	p=np.zeros(m)
	for i in range(m):
		if np.dot(X[i], theta)>=0.5:
			p[i]=1
	return p


utils.plotDecisionBoundary(plotData, theta, X, y)
plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')
plt.legend(['y = 1', 'y = 0'])
plt.grid(False)
plt.title('lambda = %0.2f' % lambda_)
plt.show()

# Compute accuracy on our training set
p = predict(theta, X)

print('Train Accuracy: %.1f %%' % (np.mean(p == y) * 100))
print('Expected accuracy (with lambda = 1): 83.1 % (approx)\n')