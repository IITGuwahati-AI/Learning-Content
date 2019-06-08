import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


print("Warm Up Exercise \nPrinting 5x5 Identity Matrix ")
print(np.eye(5))



A=np.loadtxt('ex1data1.txt',delimiter=',')
plt.scatter(A[:,0],A[:,1],s=10,c='r',marker="x")

m=np.shape(A)[0]
X=np.ones((m,2))
Y=np.zeros((m,1))
for i in range(0,m):
    X[i,1]=A[i,0]
    Y[i,0]=A[i,1]
theta=np.array([[0],[0]])
def ComputeCost(X,Y,theta):
    J_matrix=np.array(np.dot(X,theta)-Y)
    J=(1/(2*m))*np.dot(J_matrix.transpose(),J_matrix)
    return float(J)
print("The value of J calculated for theta=\n[[0]\n[0]]\n is",ComputeCost(X,Y,theta))
print("expected value for J with theta=\n[[0]\n[0]]\n is 32.07")



alpha=0.01
iterations=1500
def gradient_Descent(X,Y,theta,alpha,iters):
    for i in range(0,iters):
        B = np.zeros(np.shape(theta))
        for j in range(0,np.shape(theta)[0]):
            B[j] = np.dot((X[:, j]),(np.dot(X,theta)-Y))
        theta = np.subtract(theta,((alpha/m)*B))
    return theta
theta=gradient_Descent(X,Y,theta,alpha,iterations)
print('\n\n\n\n\n',theta)
print('expected value of theta is\n[[-3.6303]\n[ 1.1664]]\n\n')
plt.plot(X[:,1],np.dot(X,theta))
plt.show()



theta0= np.linspace(-10, 10, 100)
theta1= np.linspace(-1, 4, 100)
J=np.zeros((np.shape(theta0)[0],np.shape(theta1)[0]))
for i, t0 in enumerate(theta0):
    for j, t1 in enumerate(theta1):
        J[i, j] = ComputeCost(X, Y, np.array([[t0], [t1]]))
J = J.T
fig = plt.figure(figsize=(12, 5))
ax = fig.add_subplot(121, projection='3d')
ax.plot_surface(theta0, theta1, J, cmap='viridis')
plt.xlabel('theta0')
plt.ylabel('theta1')
plt.title('Surface')
plt.show()

#  Multi Features 


data = np.array(np.loadtxt('ex1data2.txt',delimiter=','))
X = data[:, (0,1)]
m=np.shape(X)[0]
Y=np.zeros((m,1))
for i in range(0,m):
    Y[i,0]= data[:, 2][i]


mean=np.ones((m,2))
for i in range(0,m):
    mean[i,0]=mean[i,0]*np.mean(X[:,0])
    mean[i,1]=mean[i,1]*np.mean(X[:,1])
X_norm=X-mean
for i in range(0,m):
    X_norm[i,0]=(X_norm[i,0]/np.std(X[:,0]))
    X_norm[i, 1] = (X_norm[i, 1] / np.std(X[:, 1]))
X=np.ones((m,3))
for i in range(0,m):
    X[i,0]=1
    X[i,1]=X_norm[i,0]
    X[i,2]=X_norm[i,1]
def ComputeCostMulti(X, Y, theta):
    J=np.dot(((np.dot(X,theta)-Y).transpose()),(np.dot(X,theta)-Y))
    return J/(2*m)
alpha = 0.01
num_iters = 600
theta = np.zeros((3, 1))
def gradientDescentMulti(X, y, theta, alpha, num_iters):
    J_history = np.zeros((num_iters, 1))
    for i in range(0,num_iters):
        B = np.zeros(np.shape(theta))
        for j in range(0,np.shape(theta)[0]):
            B[j] = np.dot((X[:, j]),(np.dot(X,theta)-Y))
        theta = np.subtract(theta,((alpha/m)*B))
        J_history[i]=ComputeCostMulti(X,Y,theta)
    return [theta,J_history]
[theta, J_history] = gradientDescentMulti(X, Y, theta, alpha, num_iters)
print(theta)
plt.plot(J_history)
plt.xlabel('No. of iterations')
plt.ylabel('Cost function J')
plt.show()


a=[1650,3]
a[0]=a[0]-np.mean(X[:,1])
a[0]=a[0]/np.std(X[:,1])

a[1]=a[1]-np.mean(X[:,2])
a[1]=a[1]/np.std(X[:,2])
b=np.array([1,a[0],a[1]])
price=np.dot(b,theta)
print('Price predicted from gradient descent=',price)


X = data[:, (0,1)]
K=X
X=np.ones((m,3))
for i in range(0,m):
    X[i,0]=1
    X[i,1]=K[i,0]
    X[i,2]=K[i,1]
theta=np.dot(np.dot(np.linalg.inv(np.dot(X.transpose(),X)),X.transpose()),Y)
print(theta)

x=[1,1650,3]
price=np.dot(x,theta)
print("Price predicted from normal method=",price)

