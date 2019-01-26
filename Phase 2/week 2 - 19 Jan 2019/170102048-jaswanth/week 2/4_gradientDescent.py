import numpy as np
import matplotlib.pyplot as plt

data=np.loadtxt(open('./ex1data1.txt','rb'),delimiter=',')
#print(data)
X=data[:,0]
Y=data[:,1]
m=np.size(Y)
#print(data.shape,m)

X=X.reshape(np.size(X),1)
Y=Y.reshape(np.size(Y),1)
X=np.insert(X,0,1,axis=1)
#print(X)
theta = np.zeros((2,1),dtype=np.float)
iterations = 1500
alpha=0.01


def computeCost(x,y,theta):
    h_theta = np.matmul(x,theta);
    error_square = np.power((h_theta-y),2);
    return np.sum(error_square)/(2*np.size(h_theta))


print('running gradient descent....\n')
print('enter the theta value to find the local minima : ')
input('enter theta0:')
input('enter theta1:')

def findThetaLocalMin(x,y,theta,iterations,alpha):
    for i in range(1,iterations):
        h_theta = np.matmul(x,theta);
        theta[0,0]=theta[0,0]-(alpha/97)*np.sum(np.multiply((h_theta-y).T,x[:,0].reshape(97,1).T).reshape(97,1))
        theta[1,0]=theta[1,0]-(alpha/97)*np.sum(np.multiply((h_theta-y).T,x[:,1].reshape(97,1).T).reshape(97,1))
    return theta

theta = findThetaLocalMin(X,Y,theta,iterations,alpha)

print('Local minima of cost function is at :\t'+ str(theta));
print('minimum cost is : '+ str(computeCost(X,Y,theta)))