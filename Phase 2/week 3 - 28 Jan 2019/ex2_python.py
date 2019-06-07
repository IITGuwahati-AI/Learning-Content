import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize
import utils
#function to plot the data

def plotData(X, y):
    fig = plt.figure()
    pos = np.where(y==0)
    neg = np.where(y==1)
    plt.scatter(X[pos,0],X[pos,1])
    plt.scatter(X[neg,0],X[neg,1])
#sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
'''def sigmoid(z):
    z = np.array(z)
    g = np.zeros(z.shape)
    (m, n) = z.shape
    for i in range(m):
        for j in range(n):
            g[i,j] = 1/(1+np.exp(-z[i,j]))
    return g'''          
#Costfunction and the Gradient
def costFunction(theta, X, y):
    # Initialize some useful values
    m = y.size  # number of training examples
                         
    # You need to return the following variables correctly 
    J = 0                
    grad = np.zeros(theta.shape)
    theta = theta.reshape(n+1,1)
    z = X.dot(theta)  
    h = sigmoid(z)       
    y_dash = y.reshape(1,100)
    y = y.reshape(100,1)     
    J = (-y_dash.dot(np.log(h)) - (1 - y_dash).dot(np.log(1 - h)))/m
    J = J.reshape(1,)
    grad = np.dot(X.T, (h - y))/m
    grad = grad.reshape(3,)
    return J,grad   
#predict function 
def predict(theta, X):
    m = X.shape[0] # Number of training examples

    # You need to return the following variables correctly
    p = np.zeros(m)
    theta = theta.reshape(n+1,1)
    z = X.dot(theta)  
    h = sigmoid(z)    
    for i in range(m):
        if h[i]>=0.5:
            p[i] = 1
    return p       
# Load data
# The first two columns contains the exam scores and the third column
# contains the label.
data = np.loadtxt('ex2data1.txt', delimiter=',')
X, y = data[:, 0:2], data[:, 2]   


plotData(X, y)
# add axes labels
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.legend(['Admitted', 'Not admitted'])
pass


# Setup the data matrix appropriately, and add ones for the intercept term
m, n = X.shape
# Add intercept term to X
X = np.concatenate([np.ones((m, 1)), X], axis=1)
# Initialize fitting parameters
initial_theta = np.zeros(n+1)

cost, grad = costFunction(initial_theta, X, y)

print('Cost at initial theta (zeros): {:.3f}'.format(*cost))
print('Expected cost (approx): 0.693\n')

print('Gradient at initial theta (zeros):')
print('\t[{:.4f}, {:.4f}, {:.4f}]'.format(*grad))
print('Expected gradients (approx):\n\t[-0.1000, -12.0092, -11.2628]\n')

# Compute and display cost and gradient with non-zero theta
test_theta = np.array([-24, 0.2, 0.2])
cost, grad = costFunction(test_theta, X, y)

print('Cost at test theta: {:.3f}'.format(*cost))
print('Expected cost (approx): 0.218\n')

print('Gradient at test theta:')
print('\t[{:.3f}, {:.3f}, {:.3f}]'.format(*grad))
print('Expected gradients (approx):\n\t[0.043, 2.566, 2.647]')



# set options for optimize.minimize
options= {'maxiter': 400}

# see documention for scipy's optimize.minimize  for description about
# the different parameters
# The function returns an object `OptimizeResult`
# We use truncated Newton algorithm for optimization which is 
# equivalent to MATLAB's fminunc
# See https://stackoverflow.com/questions/18801002/fminunc-alternate-in-numpy
res = optimize.minimize(costFunction,
                        initial_theta,
                        (X, y),
                        jac=True,
                        method='TNC',
                        options=options)

# the fun property of `OptimizeResult` object returns
# the value of costFunction at optimized theta
cost = res.fun

# the optimized theta is in the x property
theta = res.x

# Print theta to screen
print('Cost at theta found by optimize.minimize: {:.3f}'.format(*cost))
print('Expected cost (approx): 0.203\n');

print('theta:')
print('\t[{:.3f}, {:.3f}, {:.3f}]'.format(*theta))
print('Expected theta (approx):\n\t[-25.161, 0.206, 0.201]')

# Plot Boundary
utils.plotDecisionBoundary(plotData, theta, X, y)
#  Predict probability for a student with score 45 on exam 1 
#  and score 85 on exam 2 
prob = sigmoid(np.dot([1, 45, 85], theta))
print('For a student with scores 45 and 85,'
      'we predict an admission probability of {:.3f}'.format(prob))
print('Expected value: 0.775 +/- 0.002\n')

# Compute accuracy on our training set
p = predict(theta, X)
print('Train Accuracy: {:.2f} %'.format(np.mean(p == y) * 100))
print('Expected accuracy (approx): 89.00 %')