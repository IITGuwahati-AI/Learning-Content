import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize
import utils
#function to plot the data
def plotData(X, y):
    pos = np.where(y==0)
    neg = np.where(y==1)
    plt.scatter(X[pos,0],X[pos,1])
    plt.scatter(X[neg,0],X[neg,1])
#sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
#regilaraised cost function
def costFunctionReg(theta, X, y, lambda_):
    # Initialize some useful values
    m = y.size  # number of training examples

    # You need to return the following variables correctly 
    J = 0
    grad = np.zeros(theta.shape)
    theta = theta.reshape(theta.shape[0],1)
    z = X.dot(theta)  
    h = sigmoid(z)       
    y_dash = y.reshape(1,m)
    y = y.reshape(m,1)     
    J = (-y_dash.dot(np.log(h)) - (1 - y_dash).dot(np.log(1 - h)))/m + ((theta.T).dot(theta)-theta[0]**2)*lambda_/(2*m)
    J = J.reshape(1,)
    extra = np.zeros(theta.shape)
    extra[0] = theta[0,0]
    grad = np.dot(X.T, (h - y))/m + theta*lambda_/m - extra*lambda_/m
    grad = grad.reshape(28,)
    return J,grad   
#predict function 
def predict(theta, X):
    m = X.shape[0] # Number of training examples

    # You need to return the following variables correctly
    p = np.zeros(m)
    theta = theta.reshape(theta.shape[0],1)
    z = X.dot(theta)  
    h = sigmoid(z)    
    for i in range(m):
        if h[i]>=0.5:
            p[i] = 1
    return p       

# Load Data
# The first two columns contains the X values and the third column
# contains the label (y).
data = np.loadtxt('ex2data2.txt', delimiter=',')
X = data[:, :2]
y = data[:, 2]
#plotting the given data

plotData(X, y)
# Labels and Legend
plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')

# Specified in plot order
plt.legend(['y = 1', 'y = 0'], loc='upper right')
pass
# Note that mapFeature also adds a column of ones for us, so the intercept
# term is handled
X = utils.mapFeature(X[:, 0], X[:, 1])

# Initialize fitting parameters
initial_theta = np.zeros(X.shape[1])

# Set regularization parameter lambda to 1
# DO NOT use `lambda` as a variable name in python
# because it is a python keyword
lambda_ = 1

# Compute and display initial cost and gradient for regularized logistic
# regression
cost, grad = costFunctionReg(initial_theta, X, y, lambda_)

print('Cost at initial theta (zeros): {:.3f}'.format(*cost))
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
print('Cost at test theta    : {:.2f}'.format(*cost))
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

utils.plotDecisionBoundary(plotData, theta, X, y)
plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')
plt.legend(['y = 1', 'y = 0'])
plt.grid(False)
plt.title('lambda = %0.2f' % lambda_)

# Compute accuracy on our training set
p = predict(theta, X)

print('Train Accuracy: %.1f %%' % (np.mean(p == y) * 100))
print('Expected accuracy (with lambda = 1): 83.1 % (approx)\n')