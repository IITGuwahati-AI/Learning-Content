import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("/home/abhishek/ex1data1.txt", delimiter=',')
m=len(data)
p=np.size(data,1)
alpha = 0.01
iterations = 1500
theta = np.zeros((p,1))


y = data[:,-1:]
X = np.ones((m,p))
X[:,1:] = data[:,0:p-1]

for i in range(0,iterations):
     cost = np.dot(X,theta) - y 
     G = np.dot(X.T,cost)/m
     theta -= alpha*G

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

