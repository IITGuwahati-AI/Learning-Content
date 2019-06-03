# used for manipulating directory paths
import os

# Scientific and vector computation for python
import numpy as np

# Plotting library
from matplotlib import pyplot

# Optimization module in scipy
from scipy import optimize
import utils

data = np.loadtxt(os.path.join("../Data/ex2data1.txt"), delimiter=',')
X, y = data[:, 0:2], data[:, 2]
pos = y == 1
neg = y == 0

# Plot Examples
pyplot.plot(X[pos, 0], X[pos, 1], 'k*', lw=2, ms=10)
pyplot.plot(X[neg, 0], X[neg, 1], 'ko', mfc='y', ms=8, mec='k', mew=1)

def plotData(X, y):
    fig=pyplot.figure()
    pyplot.scatter(X[:,0][y==0],X[:,1][y==0],label='Not Admitted')
    pyplot.scatter(X[:,0][y==1],X[:,1][y==1],label='Admitted' )
    pyplot.title('Scatter plot of Training Data')
plotData(X, y)
# add axes labels
pyplot.xlabel('Exam 1 score')
pyplot.ylabel('Exam 2 score')
pyplot.legend(['Admitted', 'Not admitted'])
pass

def sigmoid(z):
    z=np.array(z)
    g=np.zeros(z.shape)
    g=1+np.exp(-z)
    g=1/g
    return g
z = 0
g = sigmoid(z)

print('g(', z, ') = ', g)
m, n = X.shape

# Add intercept term to X
X = np.concatenate([np.ones((m, 1)), X], axis=1)
def costFunction(theta, X, y):
    m = y.size
    J = 0
    grad = np.zeros(theta.shape)
    k=np.dot(X,theta)
    k=sigmoid(k)
    J=np.log(k)
    J=np.dot(J.T,y)
    p=np.log(1-k)
    p=np.dot(p.T,1-y)
    J=-J-p
    J=J/m
    grad=k-y
    grad=np.dot(X.T,grad)
    grad=grad/m
    return J,grad
def costFunction(theta, X, y):
    m = y.size
    J = 0
    grad = np.zeros(theta.shape)
    k=np.dot(X,theta)
    k=sigmoid(k)
    J=np.log(k)
    J=np.dot(J.T,y)
    p=np.log(1-k)
    p=np.dot(p.T,1-y)
    J=-J-p
    J=J/m
    grad=k-y
    grad=np.dot(X.T,grad)
    grad=grad/m
    return J,grad
initial_theta = np.zeros(n+1)

cost, grad = costFunction(initial_theta, X, y)

print('Cost at initial theta (zeros): {:.3f}'.format(cost))
print('Expected cost (approx): 0.693\n')

print('Gradient at initial theta (zeros):')
print('\t[{:.4f}, {:.4f}, {:.4f}]'.format(*grad))
print('Expected gradients (approx):\n\t[-0.1000, -12.0092, -11.2628]\n')

# Compute and display cost and gradient with non-zero theta
test_theta = np.array([-24, 0.2, 0.2])
cost, grad = costFunction(test_theta, X, y)

print('Cost at test theta: {:.3f}'.format(cost))
print('Expected cost (approx): 0.218\n')

print('Gradient at test theta:')
print('\t[{:.3f}, {:.3f}, {:.3f}]'.format(*grad))
print('Expected gradients (approx):\n\t[0.043, 2.566, 2.647]')
options= {'maxiter': 400}
res = optimize.minimize(costFunction,initial_theta,(X,y),jac=True,method='TNC',options=options)
cost = res.fun
theta=res.x
print('Cost at theta found by optimize.minimize: {:.3f}'.format(cost))
print('Expected cost (approx): 0.203\n');

print('theta:')
print('\t[{:.3f}, {:.3f}, {:.3f}]'.format(*theta))
print('Expected theta (approx):\n\t[-25.161, 0.206, 0.201]')
utils.plotDecisionBoundary(plotData, theta, X, y)
def predict(theta, X):
     m = X.shape[0]
     p=np.zeros(m)
     for i in range(m):
         if(sigmoid(np.dot(X[i,:],theta))>=0.5):
             p[i]=1
         else:
             p[i]=0
     return p        
prob = sigmoid(np.dot([1, 45, 85], theta))
print('For a student with scores 45 and 85,'
      'we predict an admission probability of {:.3f}'.format(prob))
print('Expected value: 0.775 +/- 0.002\n')

# Compute accuracy on our training set
p = predict(theta, X)
print('Train Accuracy: {:.2f} %'.format(np.mean(p == y) * 100))
print('Expected accuracy (approx): 89.00 %') 
data = np.loadtxt(os.path.join("../Data/ex2data2.txt"), delimiter=',')
X, y = data[:, 0:2], data[:, 2]
pos = y == 1
neg = y == 0
def plotData(X, y):
    fig=pyplot.figure()
    pyplot.plot(X[pos, 0], X[pos, 1], 'k*', lw=2, ms=10)
    pyplot.plot(X[neg, 0], X[neg, 1], 'ko', mfc='y', ms=8, mec='k', mew=1)
plotData(X, y)
# Labels and Legend
pyplot.xlabel('Microchip Test 1')
pyplot.ylabel('Microchip Test 2')

# Specified in plot order
pyplot.legend(['y = 1', 'y = 0'], loc='upper right')
pass
X = utils.mapFeature(X[:, 0], X[:, 1])
def costFunctionReg(theta, X, y, lambda_):
    m = y.size
    J = 0
    grad = np.zeros(theta.shape)
    k=np.dot(X,theta)
    k=sigmoid(k)
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

utils.plotDecisionBoundary(plotData, theta, X, y)
pyplot.xlabel('Microchip Test 1')
pyplot.ylabel('Microchip Test 2')
pyplot.legend(['y = 1', 'y = 0'])
pyplot.grid(False)
pyplot.title('lambda = %0.2f' % lambda_)

# Compute accuracy on our training set
p = predict(theta, X)

print('Train Accuracy: %.1f %%' % (np.mean(p == y) * 100))
print('Expected accuracy (with lambda = 1): 83.1 % (approx)\n')

    