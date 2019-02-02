import numpy as np
import matplotlib.pyplot as plt 
import math
import scipy.optimize as op

data=np.loadtxt('./ex2data1.txt',delimiter=",");
x=data[:,[0,1]]
y=data[:,2]
[m,n]=x.shape
x=np.insert(x,0,1,axis=1)
initial_theta=np.zeros(n+1)
y=y.reshape(np.size(y),1)

def sigmoidFunction(x):
	return 1/(1+np.exp(-x))

def hTheta(initial_theta,x):
	# print(np.matmul(x,initial_theta).shape,'yes')
	# print(np.matmul(x,initial_theta).shape)
	return sigmoidFunction(np.matmul(x,initial_theta))

def costFunction(initial_theta,x,y):
	r1=np.matmul(np.log(hTheta(initial_theta,x)),y)
	r1+=np.matmul(np.log((1-hTheta(initial_theta,x))),1-y)
	r2=np.matmul(x.T,(hTheta(initial_theta,x).reshape(len(y),1)-y))
	return -r1/m,r2

cost,gradient=costFunction(initial_theta,x,y)

def optMinimum(initial_theta,x,y):
	options= {'maxiter': 400}
	r=op.minimize(costFunction,initial_theta,(x,y),method='TNC' ,jac=True,options=options)
	return r.x

theta=optMinimum(initial_theta,x,y)
[cost,gradient]=costFunction(optMinimum(initial_theta,x,y),x,y) 

def predict(theta,x):
	p=np.zeros(np.size(x[:,0]),dtype=int);
	p=sigmoidFunction(np.matmul(x,theta))
	p=p>=0.5
	return p*1

print('give values to predict')
n=input('no.of training examples : ')
n=int(n)
examples=np.ones((n,3),dtype=float)
print('enter...')
for i in range(0,n):
	examples[i,1]=input('exam1 score:')
	examples[i,2]=input('exam2 score:')

theta=theta.reshape(3,1)
p=predict(theta,examples)
print(p)