import os
import numpy as np
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D

def warmUpExercise():
	A=np.eye(5)
	return A;
warmUpExercise()

data = np.loadtxt("../Data/ex1data1.txt", delimiter=',')
X, y = data[:, 0], data[:, 1]
m = y.size


def plotData(x,y):
	fig=pyplot.figure()
	pyplot.plot(x,y,'ro',ms=10,mec='k')


plotData(X,y)
pyplot.show()


X=np.stack([np.ones(m),X],axis=1)


def computeCost(X,y,theta):
	m=y.size
	J=0;
	k=np.dot(X,theta)-y
	k=k**2
	k=sum(k)
	J=k/(2*m)
	return J

J = computeCost(X, y, theta=np.array([0.0, 0.0]))
print('With theta = [0, 0] \nCost computed = %.2f' % J)
print('Expected cost value (approximately) 32.07\n')

# further testing of the cost function
J = computeCost(X, y, theta=np.array([-1, 2]))
print('With theta = [-1, 2]\nCost computed = %.2f' % J)
print('Expected cost value (approximately) 54.24')


def gradientDescent(X,y,theta,alpha,num_iters):
	m=y.shape[0]
	theta=theta.copy()
	J_history=[]
	for i in range(num_iters):
		theta=theta-(alpha/m)*(X.T@(X@(theta)-y))
		cost=computeCost(X,y,theta)
		J_history.append(cost)
	return theta, J_history

# initialize fitting parameters
theta = np.zeros(2)

# some gradient descent settings
iterations = 1500
alpha = 0.01

theta, J_history = gradientDescent(X ,y, theta, alpha, iterations)
print('Theta found by gradient descent: {:.4f}, {:.4f}'.format(*theta))
print('Expected theta values (approximately): [-3.6303, 1.1664]')

# plot the linear fit
plotData(X[:, 1], y)
pyplot.plot(X[:, 1], np.dot(X, theta), '-')
pyplot.legend(['Training data', 'Linear regression'])
pyplot.show()
# Predict values for population sizes of 35,000 and 70,000
predict1 = np.dot([1, 3.5], theta)
print('For population = 35,000, we predict a profit of {:.2f}\n'.format(predict1*10000))

predict2 = np.dot([1, 7], theta)
print('For population = 70,000, we predict a profit of {:.2f}\n'.format(predict2*10000))

# grid over which we will calculate J
theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)

# initialize J_vals to a matrix of 0's
J_vals = np.zeros((theta0_vals.shape[0], theta1_vals.shape[0]))

# Fill out J_vals
for i, theta0 in enumerate(theta0_vals):
	for j, theta1 in enumerate(theta1_vals):
		J_vals[i, j] = computeCost(X, y, [theta0, theta1])
# Because of the way meshgrids work in the surf command, we need to
# transpose J_vals before calling surf, or else the axes will be flipped
J_vals = J_vals.T

# surface plot
fig = pyplot.figure(figsize=(12, 5))
ax = fig.add_subplot(121, projection='3d')
ax.plot_surface(theta0_vals, theta1_vals, J_vals, cmap='viridis')
pyplot.xlabel('theta0')
pyplot.ylabel('theta1')
pyplot.title('Surface')
pyplot.show()

# contour plot
# Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
ax = pyplot.subplot(122)
pyplot.contour(theta0_vals, theta1_vals, J_vals, linewidths=2, cmap='viridis', levels=np.logspace(-2, 3, 20))
pyplot.xlabel('theta0')
pyplot.ylabel('theta1')
pyplot.plot(theta[0], theta[1], 'bo', ms=5, lw=2)
pyplot.title('Contour, showing minimum')
pyplot.show()
pass


#for multple features
data=np.loadtxt("../Data/ex1data2.txt",delimiter=",")
X = data[:, :2]
y = data[:, 2]
m = y.size

# print out some data points
print('{:>8s}{:>8s}{:>10s}'.format('X[:,', 'X[:, 1]', 'y'))
print('-'*26)
for i in range(10):
	print('{:8.0f}{:8.0f}{:10.0f}'.format(X[i, 0], X[i, 1], y[i]))


def featureNormalize(X):
	X_norm = X.copy()
	mu = np.zeros(X.shape[1])
	sigma = np.zeros(X.shape[1])
	for i in range(len(X[0,:])):
		mu[i]=np.mean(X[:,i])
		sigma[i]=np.std(X[:,i])
		X_norm[:,i]=(X_norm[:,i]-mu[i])/sigma[i]
	return X_norm,mu,sigma

# call featureNormalize on the loaded data
X_norm, mu, sigma = featureNormalize(X)

print('Computed mean:', mu)
print('Computed standard deviation:', sigma)

# Add intercept term to X
X = np.concatenate([np.ones((m, 1)), X_norm], axis=1)


def computeCostMulti(X,y,theta):
	m = y.shape[0]
	J=0
	k=np.dot(X,theta)-y
	k=k**2
	k=sum(k)
	J=k/(2*m)
	return J

def gradientDescentMulti(X,y,theta,alpha,num_iters):
	m=y.shape[0]
	theta=theta.copy()
	J_history=[]
	for i in range(num_iters):
		theta=theta-(alpha/m)*(X.T@(X@(theta)-y))
		J_history.append(computeCostMulti(X,y,theta))
	return theta,J_history


alpha = .03
num_iters = 400

# in it theta and run gradient descent
theta = np.zeros(3)
theta, J_history = gradientDescentMulti(X, y, theta, alpha, num_iters)

# Plot the convergence graph
pyplot.plot(np.arange(len(J_history)), J_history, lw=2)
pyplot.xlabel('Number of iterations')
pyplot.ylabel('Cost J')
pyplot.show()

# Display the gradient descent's result
print('theta computed from gradient descent: {:s}'.format(str(theta)))

price = (theta.T)@(np.array([1,1650, 3])) 

print('Predicted price of a 1650 sq-ft, 3 br house (using gradient descent): ${:.0f} \n'.format(price))

#normal equation

# Load data
data = np.loadtxt("../Data/ex1data2.txt", delimiter=',')
X = data[:, :2]
y = data[:, 2]
m = y.size
X = np.concatenate([np.ones((m, 1)), X], axis=1)

def normalEqn(X,y):
	theta = np.zeros(X.shape[1])
	theta=(np.linalg.pinv(X.T@X))@(X.T)@y
	return theta

theta = normalEqn(X, y);
print('Theta computed from the normal equations: {:s}'.format(str(theta)));
price = np.dot([1,1650,3],theta.T)
print('Predicted price of a 1650 sq-ft, 3 br house (using normal equations): ${:.0f}'.format(price))
