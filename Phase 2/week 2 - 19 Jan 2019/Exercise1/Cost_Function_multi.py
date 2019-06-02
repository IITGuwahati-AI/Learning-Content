import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("/home/abhishek/ex1data2.txt", delimiter=',')
m=len(data)
theta=np.array([-3.63029144,1.16636235])
theta = theta.T
p=np.size(data,1)

max= np.amax(data, axis=0)
min= np.amin(data, axis=0)

sum = np.sum(data,axis=0)

for i in range(0,p):
    for j in range(0, m):
        data[j,i] -= sum[i]
        data[j,i] /= (max[i] - min[i])

data = data/m
y = data[:,-1:]

X = np.ones((m,2))
X[:,1:] = data[:,0:1]
cost = np.dot(X,theta) - y
cost = cost*cost
sq= np.sum(cost)

J = sq/(2*m)
print(J)
