import os
import numpy as np
from matplotlib import pyplot as plt
import math
from scipy import optimize
import utils

data=np.genfromtxt("Data/ex2data1.txt",delimiter=',')
X,y =data[:,:2],data[:,2]
X=np.concatenate((np.array([1.0 for _ in range(y.shape[0])])[:,np.newaxis],X),axis=1)
m,n=X.shape


def plotData(X,y):
	fig=plt.figure()
	pos = y == 1
	neg = y == 0
	plt.plot(X[pos, 0], X[pos, 1], 'k*', lw=2, ms=10)
	plt.plot(X[neg, 0], X[neg, 1], 'ko', mfc='y', ms=8, mec='k', mew=1)
	plt.xlabel('Exam 1 score')
	plt.ylabel('Exam 2 score')
	plt.legend(['admitted','Not admitted'])
	


def sigmoid(z):
	return 1/(1+math.exp(-z))


def costFunction(theta,X,y):
	m=y.size
	J=0
	grad=np.zeros(theta.shape)
	for i in range(m):
		J-=((y[i]*np.log(sigmoid(np.matmul(theta.T,X[i]))))+(1-y[i])*np.log(1-sigmoid(np.matmul(theta.T,X[i]))))
		grad+=((sigmoid(np.matmul(theta.T,X[i]))-y[i])*X[i])
	J/=m;grad/=m
	return J,grad


initial_theta=np.zeros(n)
options={'maxiter': 400}
res=optimize.minimize(costFunction,initial_theta,(X,y),jac=True,method='TNC',options=options)
cost = res.fun
theta= res.x

print('Cost at theta found by optimize.minimize: {:.3f}'.format(cost))
print('Expected cost (approx): 0.203\n');

print('theta:')
print('\t[{:.3f}, {:.3f}, {:.3f}]'.format(*theta))
print('Expected theta (approx):\n\t[-25.161, 0.206, 0.201]')

utils.plotDecisionBoundary(plotData, theta, X, y)
plt.show()
def predict(theta,X):
	m=X.shape[0]
	p=np.zeros(m)
	for i in range(m):
		if np.dot(X[i], theta)>=0.5:
			p[i]=1
	return p


p=predict(theta,X)
print('Train Accuracy: {:.2f} %'.format(np.mean(p == y) * 100))
print('Expected accuracy (approx): 89.00 %')