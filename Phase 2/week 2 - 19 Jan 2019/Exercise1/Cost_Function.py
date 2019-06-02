import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("/home/abhishek/ex1data1.txt", delimiter=',')
m=len(data)
theta = np.zeros((2,1))
y = data[:,-1:]

X = np.ones((m,2))
X[:,1:] = data[:,0:1]
cost = np.dot(X,theta) - y
cost = cost*cost
sq= np.sum(cost)

J = sq/(2*m)
print(J)


