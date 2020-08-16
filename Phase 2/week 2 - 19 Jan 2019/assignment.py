import os
import matpoltlib as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
def warmUpExercise():
    A=np.eye(5)
    return A
warmUpExercise()    
data = np.loadtxt('ex1data1.txt' ,delimiter=',')
x1=data[:,0]
y=data[:,1]
m=y.size
def plotData(x,y):
    pyplot.plot(x,y,'ro',ms=10,mec='k')
    pyplot.ylabel('profit in $10,000')
    pyplot.xlabel('population of city in 10,000s')
plotData(x1,y)
x1 = np.stack([np.ones(m),x1],axis=1)


def computeCost(x,y,theta):
    m=y.size
    c=0
    c=np.dot(x,theta)-y
    c=(np.dot(c,c))/(2*m)
    return c
C = computeCost(x1,y,theta=np.array([0.00,0.00]))
print('expected cost function at theta=[0,0] cost=%.2f'C)

C = computeCost(x1,y,theta=np.array([-1,2]))
print('expected cost function at theta=[-1,2]=%.2f'C)


def gradientDescent(theta,x,y,a,n):
	m=y.shape[0]
	theta=theta.copy()
	C_history=[]
	for i in range m:
		theta=theta.copy()-a*np.dot(x.T,np.dot(x,theta)-y)
		C_history.append(computeCost(x,y,theta))
	return theta,C_history	


theta=np.array([0,0])
n=1500
a=0.01
theta,C_history=gradientDescent(theta,x1,y,a,n)


plotData(x1[:,1],y)
pyplot.plot(x1[:,1],np.dot(x1,theta),'-')
pyplot.legend(['training data','linear regresion'])

theta0=np.linspace(-10,10,100)
theta1=np.linspace(-1,4,100)
j_values = np.zeros(theta0.shape[0],theta1.shape[1])
for i,theta_0 in enumerate(theta0):
	for j ,theta_1 in enumerate(theta1):
		j_values[i,j]=computeCost(x1,y,[theta_0,theta_1])
j_values=j_values.T


fig=pyplot.figure(figsize=(12,5))
ax=fig.add_subplot(121,projection='3d')
ax.plot_surface(theta0,theta1,j_values,linewidths=2,cmap='viridis',levels=np.logspace(-2,3,20))
pyplot.xlabel('theta0')
pyplot.ylabel('theta1')
pyplot.title('surface')
ax=pyplot.subplot(122)	
pyplot.contour(theta0,theta1,j_values,linewidths=2,cmap='viridis',levels=np.logspace(-2,3,20))
pyplot.xlabel('theta0')
pyplot.ylabel('theta1')
pyplot.plot(theta[0],theta[1],'ro',ms=10,lw=2)
pyplot.title('contour,showing min')

pass
data=mp.loadtxt(, delimiter=',')
X=data[:,:2]
y=data[:,:2]
m=y.size
print('{:>8s}{:>8s}{:>10s}'.format('X[:,0]', 'X[:, 1]', 'y'))
print('-'*26)
for i in range(10):
    print('{:8.0f}{:8.0f}{:10.0f}'.format(X[i, 0], X[i, 1], y[i]))
def  featureNormalize(X):
    X_norm = X.copy()
    mu = np.zeros(X.shape[1])
    sigma = np.zeros(X.shape[1])
    m1=np.mean(X_norm,axis=0)
    mu=mu+m1
    s1=np.std(X_norm,axis=0)
    sigma=s1+sigma
    for k in range(0,X.shape[1]):
        X_norm[:,k]= (X[:,k]-np.ones(X.shape[0])*mu[k])/sigma[k]

    return X_norm,mu,sigma  
X_norm, mu, sigma = featureNormalize(X)
print('Computed mean:', mu)
print('Computed standard deviation:', sigma)
X = np.concatenate([np.ones((m, 1)), X_norm], axis=1)
def computeCostMulti(X, y, theta):
    m = y.shape[0]
    J=0
    J=np.dot(X,theta)-y
    J=np.dot(J,J)
    J=J/2
    J=J/m
    return J
def gradientDescentMulti(X, y, theta, alpha, num_iters):

    m = y.shape[0]
    theta = theta.copy()
    J_history = []
    for i in range(num_iters):
        theta=theta.copy()
        theta=theta.copy()-alpha*np.dot(X.T,np.dot(X,theta)-y)/m
        J_history.append(computeCost(X,y,theta))
    return theta,J_history
alpha = 0.1
num_iters = 400

# init theta and run gradient descent
theta = np.zeros(3)
theta, J_history = gradientDescentMulti(X, y, theta, alpha, num_iters)

# Plot the convergence graph
pyplot.plot(np.arange(len(J_history)), J_history, lw=2)
pyplot.xlabel('Number of iterations')
pyplot.ylabel('Cost J')

# Display the gradient descent's result
print('theta computed from gradient descent: {:s}'.format(str(theta)))
price = 0
print('Predicted price of a 1650 sq-ft, 3 br house (using gradient descent): ${:.0f}'.format(price))
data = np.loadtxt('../Data/ex1data2.txt', delimiter=',')
X = data[:, :2]
y = data[:, 2]
m = y.size
X = np.concatenate([np.ones((m, 1)), X], axis=1)
def normalEqn(X, y):
    theta = np.zeros(X.shape[1])
    theta1=np.dot(X.T,X)
    theta1=np.linalg.pinv(theta1)
    theta1=np.dot(theta1,X.T)
    theta=np.dot(theta1,y)
    return theta
theta = normalEqn(X, y);
print('Theta computed from the normal equations: {:s}'.format(str(theta)));
price = np.dot([1,1650,3],theta)
print('Predicted price of a 1650 sq-ft, 3 br house (using normal equations): ${:.0f}'.format(price))










