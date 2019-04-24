import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
data=np.genfromtxt("ex1data2.txt",delimiter=',')
fig=plt.figure()
gr=Axes3D(fig)
gr.scatter(xs=data[:,0], ys=data[:,1], zs=data[:,2], zdir='z', label='ys=0, zdir=z')
gr.set_xlabel('size of house')
gr.set_ylabel('no of bedrooms')
gr.set_zlabel('price of the house')
plt.show()
data[:,0]/=1000
data[:,2]/=100000

x=np.concatenate((np.array([1 for _ in range(data.shape[0])])[:,np.newaxis],data),axis=1)
theta0=1.0;theta1=1.0;theta2=1.0
theta=np.array([theta0,theta1,theta2])
def cost():
	cost_function=np.array([0.0])
	for i in range(len(data[:,0])):
		cost_function+=(np.matmul(theta,x[i,:3])-data[i,2])**2
	cost_function/=2*len(data[:,0])
	return cost_function[0]
def cost_derivative(x):
	global theta
	derivative=np.array([0.0])
	if x==2:
		for i in range(len(data[:,0])):
			derivative+=(np.matmul(theta,(np.array([1]+list(data[i,:2]))[:, np.newaxis]))-data[i,2])
		derivative/=len(data[:,0])
		return derivative[0]
	else:
		for i in range(len(data[:,0])):
			derivative+=(np.matmul(theta,(np.array([1]+list(data[i,:2]))[:, np.newaxis]))-data[i,2])*data[i,x]
		derivative/=len(data[:,0])
		return derivative[0]

def gradient_descent():
	global theta
	a=np.copy(theta)
	a[0]=a[0]-0.1*cost_derivative(2)
	a[1]=a[1]-0.1*cost_derivative(0)
	a[2]=a[2]-0.1*cost_derivative(1)
	theta=np.copy(a)
#min=cost()
k=cost()
while True:
	gradient_descent()
	p=cost()
	if round(k,20)==round(p,20):
		break
	k=p
theta[0]*=100000
theta[1]*=100
theta[2]*=100000
print(f'final value of cost function is {k*(10**10)}')
print(f'And the values of theta are {theta}')




	




