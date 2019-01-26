import numpy as np
import matplotlib.pyplot as plt

def featureNormalize(x):
	mean=np.mean(x,axis=0)
	x=x-mean
	sigma=np.std(x,axis=0)
	x[:,0]/=sigma[0]
	x[:,1]/=sigma[1]
	return x,mean,sigma
def computeCost(x,y,theta):
	cost=np.matmul((np.matmul(x,theta)-y).T,(np.matmul(x,theta)-y));
	return cost
def normalEquation(x,y,theta):
	theta=np.matmul(np.matmul(np.linalg.inv(np.matmul(x.T,x)),x.T),y)
	return theta

data=np.loadtxt(open('./ex1data2.txt','rb'),delimiter=',')
x=data[:,0:2]
y=data[:,2]
m=np.size(y)
[x,mean,sigma]=featureNormalize(x)
x=np.insert(x,0,1,axis=1)
y=y.reshape(m,1)

alpha=0.01
theta=np.zeros((3,1),dtype=float)

print('Calculating theta value using normalEquation...')
print('give the values for initiating theta : ')
theta[0,0]=input('enter theta0 : ')
theta[1,0]=input('enter theta1 : ')
theta[2,0]=input('enter theta2 : ')

theta=normalEquation(x,y,theta)
print('local minima is at')
print(theta)