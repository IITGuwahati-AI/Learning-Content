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

def gradientDescentMulti(x,y,theta,alpha,iterations):
	J_history=np.zeros((iterations,1),dtype=float)
	m=np.size(y)
	for i in range(1,iterations):
		h_theta = np.matmul(x,theta);
		theta[0,0]=theta[0,0]-(alpha/m)*np.sum(np.multiply((h_theta-y).T,x[:,0].reshape(m,1).T).reshape(m,1))
		theta[1,0]=theta[1,0]-(alpha/m)*np.sum(np.multiply((h_theta-y).T,x[:,1].reshape(m,1).T).reshape(m,1))
		theta[2,0]=theta[2,0]-(alpha/m)*np.sum(np.multiply((h_theta-y).T,x[:,2].reshape(m,1).T).reshape(m,1))
		J_history[i,0]=computeCost(x,y,theta)
	return theta,J_history

data=np.loadtxt(open('./ex1data2.txt','rb'),delimiter=',')
x=data[:,0:2]
y=data[:,2]
m=np.size(y)
[x,mean,sigma]=featureNormalize(x)
x=np.insert(x,0,1,axis=1)
y=y.reshape(m,1)

alpha=0.01
iterations=1800
theta=np.zeros((3,1),dtype=float)

print('obtaing local minima using gradient Descent algo....')

[theta, J_history] = gradientDescentMulti(x, y, theta, alpha,iterations);
print('local minima is at : ')
print(theta)
print('plot of decreasing cost value with each literation')
plt.plot(np.arange(1,iterations+1,1).reshape(iterations,1),J_history,'-b')
plt.show()
#plt.savefig('J_values')