import numpy as np
import matplotlib.pyplot as plt

data=np.loadtxt(open('./ex1data1.txt','rb'),delimiter=',')
#print(data)
X=data[:,0]
Y=data[:,1]
m=np.size(Y)
#print(data.shape,m)

def computeCost(x,y,theta):
    h_theta = np.matmul(x,theta);
    error_square = np.power((h_theta-y),2);
    return np.sum(error_square)/(2*np.size(h_theta))


X=X.reshape(np.size(X),1)
Y=Y.reshape(np.size(Y),1)
X=np.insert(X,0,1,axis=1)
#print(X)
theta = np.zeros((2,1),dtype=np.float)
iterations = 1500
alpha=0.01
print('computing cost based on given theta value')
theta[0,0]=input('enter theta0 : ')
theta[1,0]=input('enter theta1 : ')

print('Cost is '+str(computeCost(X,Y,theta)))