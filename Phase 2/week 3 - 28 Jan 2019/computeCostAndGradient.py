import numpy as np
import matplotlib.pyplot as plt 
import math
data=np.loadtxt('./ex2data1.txt',delimiter=",");
x=data[:,[0,1]]
y=data[:,2]
[m,n]=x.shape
x=np.insert(x,0,1,axis=1)
initial_theta=np.zeros((n+1,1),dtype=float)
y=y.reshape(np.size(y),1)

def sigmoidFunction(x):
	x=1+np.exp(-x)
	return np.power(x,-1)

def hTheta(initial_theta,x):
	return sigmoidFunction(np.matmul(x,initial_theta))

def costFunction(initial_theta,x,y):
	r1=np.matmul(np.log(hTheta(initial_theta,x).reshape(1,100)[0]),y)
	r1+=np.matmul(np.log((1-hTheta(initial_theta,x)).reshape(1,100)[0]),1-y)

	r2=np.matmul(x.T,hTheta(initial_theta,x)-y)
	return -r1/m,r2/m
print('cost for zeros value of theta : ')
cost,gradient=costFunction(initial_theta,x,y)
print(cost)
print('and the gradient is ...')
print(gradient[:,0])