import numpy as np
import matplotlib.pyplot as plt
data=np.genfromtxt("ex1data2.txt",delimiter=',')
x=np.array([1 for _ in range(data.shape[0])])[:,np.newaxis]
data=np.concatenate((x,data),axis=1)
x=data[:,:3]
y=data[:,3]
theta=np.matmul(np.matmul(np.linalg.inv(np.matmul(x.T,x)),x.T),y)
cost_function=np.array([0.0])
for i in range(len(data[:,0])):
	cost_function+=(np.matmul(theta,data[i,:3])-data[i,3])**2
cost_function/=2*len(data[:,0])
print(f'final value of cost function is {cost_function[0]}')
print(f'value of theta is {theta}')