import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style
style.use('ggplot')

data =np.genfromtxt('ex2data.txt',delimiter=',')
X=data[:,:2]
y=data[:,2]
m=y.size

print('{:>8s}{:>8s}{:>10s}'.format('X[:,0]', 'X[:, 1]', 'y'))
print('-'*26)
for i in range(10):
    print('{:8.0f}{:8.0f}{:10.0f}'.format(X[i, 0], X[i, 1], y[i]))

#Normalisation
def featureNormalize(X):    
    X_norm = X.copy()
    mu = np.zeros(X.shape)
    sigma = np.zeros(X.shape)
    mu=mu+(np.mean(X,axis=0))
    sigma=sigma+(np.std(X,axis=0))
    X_norm= (X_norm - mu)/ sigma
    return X_norm

X_norm=featureNormalize(X)
X=np.concatenate([np.ones((m,1)), X_norm],axis=1)
print(X)

#Cost Function
def computeCostMulti(X, y, theta):
    m=y.size
    J=0
    temp=np.dot(X,theta)
    J=(1/(2*m))*sum((y-temp)*(y-temp))
    return J

#Gradient Descent
def gradientDescentMulti(X, y, theta, alpha, num_iters):
    m = y.shape[0]
    theta = theta.copy()
    J_history = []
    for i in range(num_iters):
        temp=np.dot(X,theta)-y
        temp0=theta[0]-((alpha/m)*sum(temp))
        temp1=theta[1]-((alpha/m)*sum(temp*X[:,1]))
        temp2=theta[2]-((alpha/m)*sum(temp*X[:,2]))
        theta=np.array([temp0,temp1,temp2])
        J_history.append(computeCostMulti(X, y, theta))
    return theta,J_history

alpha = 0.1
num_iters = 400

# init theta and run gradient descent
theta = np.zeros(3)
theta, J_history = gradientDescentMulti(X, y, theta, alpha, num_iters)

plt.plot(np.arange(len(J_history)), J_history, lw=2)
plt.xlabel('Number of iterations')
plt.ylabel('Cost J')
plt.show()

print('theta computed from gradient descent: {:s}'.format(str(theta)))



