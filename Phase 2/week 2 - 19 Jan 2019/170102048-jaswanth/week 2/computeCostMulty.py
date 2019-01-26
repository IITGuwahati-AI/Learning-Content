import numpy as np
import matplotlib.pyplot as plt

def featureNormalize(x):
	mean=np.mean(x,axis=0)
	x=x-mean
	sigma=np.std(x,axis=0)
	x[:,0]/=sigma[0]
	x[:,1]/=sigma[1]
	return x,mean,sigma
def computeCostMulty(x,y,theta):
	cost=np.matmul((np.matmul(x,theta)-y).T,(np.matmul(x,theta)-y));
	return cost


data=np.loadtxt(open('./ex1data2.txt','rb'),delimiter=',')
x=data[:,0:2]
y=data[:,2]
m=np.size(y)
[x,mean,sigma]=featureNormalize(x)
x=np.insert(x,0,1,axis=1)
y=y.reshape(m,1)

alpha=0.01
iterations=400
theta=np.zeros((3,1),dtype=float)

print('calculation of cost for the given theta values ...')
theta[0,0]=input('enter theta0 : ')
theta[1,0]=input('enter theta1 : ')
theta[2,0]=input('enter theta3 : ')
J=computeCostMulty(x,y,theta)

print('The cost is '+ str(J))