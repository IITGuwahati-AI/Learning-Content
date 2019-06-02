import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv

data = np.loadtxt("/home/abhishek/ex1data1.txt", delimiter=',')
m=len(data)
p=np.size(data,1)
alpha = 0.01
iterations = 1500
theta = np.zeros((p,1))
y = data[:,-1:]


X = np.ones((m,p))
X[:,1:] = data[:,0:p-1]

theta = np.dot(np.dot(inv(np.dot(X.T,X)),X.T),y)

h= np.dot(X, theta)
plt.plot(data[:,0], data[:,1], 'rx')
plt.plot(data[:,0], h)    
plt.xlabel("Population")
plt.ylabel("Profit")
plt.legend(['Training Data', 'Linear Regression'])

print("theta=")
print(theta)
plt.savefig("Gradient Descent Hypothesis")
plt.show()

