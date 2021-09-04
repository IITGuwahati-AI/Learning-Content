import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

def warmUpExercise():
    A=np.eye(5)
    return A

print(warmUpExercise())

data=np.genfromtxt('ex1data.txt',delimiter=',')
x=data[:,0]
y=data[:,1]

m=x.size

#Plotting the data
def plotShow(x,y):
 plt.plot(x,y,'ro',ms=10,mec='k')
 plt.xlabel("Population in 10,000s:")
 plt.ylabel("Profit in $10000s")

plotShow(x,y)
plt.show()
x=np.stack([np.ones(m).T, x.T],axis=1)
print(x)

#Computing the Cost function
def computeCost(x,y,theta):
    J=0
    predictions= x.dot(theta)
    sq_errors=(predictions-y)*(predictions-y)
    J=(1/(2*m))*sum(sq_errors)
    return J
    
J = computeCost(x, y, theta=np.array([0.0, 0.0]))
print('With theta = [0, 0] \nCost computed = %.2f' % J)
print('Expected cost value (approximately) 32.07\n')

J = computeCost(x, y, theta=np.array([-1, 2]))
print('With theta = [-1, 2]\nCost computed = %.2f' % J)
print('Expected cost value (approximately) 54.24')

#Gradient Descent
def gradientDescent(X, y, theta, alpha, num_iters):
    m=y.size;
    J_history = []
    theta=theta.copy()
    for i in range(num_iters):
        temp=np.dot(X,theta)-y
        temp0=theta[0]-((alpha/m)*sum(temp))
        temp1=theta[1]-((alpha/m)*sum(temp*X[:,1]))
        theta=np.array([temp0,temp1])
        J_history.append(computeCost(X, y, theta))
    return theta,J_history

# initialize fitting parameters
theta = np.zeros(2)

# some gradient descent settings
iterations = 1500
alpha = 0.01

theta, J_history = gradientDescent(x ,y, theta, alpha, iterations)
print('Theta found by gradient descent: {:.4f}, {:.4f}'.format(*theta))
print('Expected theta values (approximately): [-3.6303, 1.1664]')

plotShow(x[:, 1], y)
plt.plot(x[:, 1], np.dot(x, theta), '-',color='blue')
plt.legend(['Training data', 'Linear regression'])
plt.show()

# grid over which we will calculate J
theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)

# initialize J_vals to a matrix of 0's
J_vals = np.zeros((theta0_vals.shape[0], theta1_vals.shape[0]))

# Fill out J_vals
for i, theta0 in enumerate(theta0_vals):
    for j, theta1 in enumerate(theta1_vals):
        J_vals[i, j] = computeCost(x, y, [theta0, theta1])
        
# Because of the way meshgrids work in the surf command, we need to
# transpose J_vals before calling surf, or else the axes will be flipped
J_vals = J_vals.T

# surface plot
fig = plt.figure(figsize=(12, 5))
ax = fig.add_subplot(121, projection='3d')
ax.plot_surface(theta0_vals, theta1_vals, J_vals, cmap='viridis')
plt.xlabel('theta0')
plt.ylabel('theta1')
plt.title('Surface')
plt.show()

# contour plot
# Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
ax = plt.subplot(122)
plt.contour(theta0_vals, theta1_vals, J_vals, linewidths=2, cmap='viridis', levels=np.logspace(-2, 3, 20))
plt.xlabel('theta0')
plt.ylabel('theta1')
plt.plot(theta[0], theta[1], 'ro', ms=10, lw=2)
plt.title('Contour, showing minimum')
plt.show()

        

