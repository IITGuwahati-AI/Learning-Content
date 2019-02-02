import numpy as np 
import matplotlib.pyplot as plt 

data=np.loadtxt('./ex2data2.txt',delimiter=',')
x=data[:,[0,1]]
y=data[:,2]
def mapFeature(X1,X2):
	degree = 6
	out = np.ones(( X1.shape[0], sum(range(degree + 2)) )) 
	curr_column = 1
	for i in range(1, degree + 1):
		for j in range(i+1):
			out[:,curr_column] = np.power(X1,i-j) * np.power(X2,j)
			curr_column += 1
	return out
x=mapFeature(x[:,0],x[:,1])

def sigmoidFunction(x):
	x=1+np.exp(-x)
	return np.power(x,-1)

def hTheta(initial_theta,x):
	return sigmoidFunction(np.matmul(x,initial_theta))

def costFunction(initial_theta,x,y,lambdaa):
	r1=np.matmul(np.log(hTheta(initial_theta,x).reshape(1,len(y))[0]),y)
	r1+=np.matmul(np.log((1-hTheta(initial_theta,x)).reshape(1,len(y))[0]),1-y)
	r1=(-r1/len(y))+(lambdaa/(2*len(y)))*np.matmul(initial_theta.T,initial_theta)

	r2=(np.matmul(x.T,hTheta(initial_theta,x)-y.reshape(len(y),1))/len(y))
	r2+=(lambdaa/len(y))*initial_theta
	return r1[0][0],r2

initial_theta=np.zeros((len(x[0,:]),1),dtype=float)
[cost,gradient]=costFunction(initial_theta,x,y,1)
print(cost,gradient)