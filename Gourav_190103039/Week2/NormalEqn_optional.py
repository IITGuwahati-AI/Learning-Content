import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

data = np.loadtxt('ex2data.txt', delimiter=',')
X = data[:, :2]
y = data[:, 2]
m = y.size
X = np.concatenate([np.ones((m, 1)), X], axis=1)

def normalEqn(X, y):
    theta = np.zeros(X.shape[1])
    theta=np.dot(np.linalg.inv(np.dot(X.T,X)),np.dot(X.T,y))
    return theta

# Calculate the parameters from the normal equation
theta = normalEqn(X, y)
print('Theta computed from the normal equations: {:s}'.format(str(theta)));
