import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize
from sklearn.preprocessing import PolynomialFeatures
def sigmoid(z):
    return 1/(1+np.exp(-z))
def costFunction(theta,x,y):
    m=y.size
    J=0
    grad=np.zeros(theta.shape)
    y=y[:,np.newaxis]
    h=sigmoid(x@theta[:,np.newaxis])
    J=(-y*np.log(h)-(1-y)*np.log(1-h)).mean()
    grad=x.T@(h-y)/m
    return J,grad[:,0]
def predict(theta,x):
    return 1*(sigmoid((x@theta[:,np.newaxis]))>=0.5)[:,0]
def costFunctionReg(theta, x,y,lambda_):
    m=y.size
    y=y[:,np.newaxis]
    J=0
    grad=np.zeros(theta.shape)
    theta=theta[:,np.newaxis]
    h=sigmoid(x2@theta)
    J=(-y*np.log(h)-(1-y)*np.log(1-h)).mean()+(lambda_/(2*m))*(theta[1:,0]**2).sum()
    grad=(x.T@(h-y)+lambda_*np.vstack(([0],theta[1:])))/m
    return J,grad[:,0]

data1=np.genfromtxt('ex2data1.txt',delimiter=",")
x=data1[:,:-1]
y=data1[:,-1]
g1=y==1
g0=y==0
plt.scatter(x[g1,0],x[g1,1],color='green')
plt.scatter(x[g0,0],x[g0,1],color='red')
plt.show()
m,n=x.shape
x=np.c_[np.ones((m,1)),x]
initial_theta= np.zeros(n+1)
cost, grad=costFunction(initial_theta,x,y)
print(cost)
print(grad)
test_theta=np.array([-24,0.2,0.2])
cost,grad=costFunction(test_theta,x,y)
print(cost)
print(grad)
options={'maxiter':400}
res=optimize.minimize(costFunction,initial_theta,(x,y),jac=True,method='TNC',options=options)
cost=res.fun
theta=res.x
print(f'Cost at optimised theta: {cost.round(3)}')
print(f'The optimised theta: {theta}')
p=predict(theta,x)
print(f'Train Accuracy: {np.mean(p==y)*100}')
print("")

#Using Regularisation
data2=np.genfromtxt('ex2data2.txt',delimiter=",")
x2=data2[:,:-1]
y2=data2[:,-1]
g1=y2==1
g0=y2==0
plt.scatter(x2[g1,0],x2[g1,1],color='green')
plt.scatter(x2[g0,0],x2[g0,1],color='red')
plt.show()
poly=PolynomialFeatures(6)
x2=poly.fit_transform(x2)
initial_theta=np.zeros(x2.shape[1])
lambda_=1
cost,grad=costFunctionReg(initial_theta,x2,y2,lambda_)
print("Theta all zeros and lambda=1")
print(f'Cost: {cost}')
print(f'Gradient(first five elements): {grad[:5].round(4)}')
test_theta=np.ones(x2.shape[1])
cost,grad=costFunctionReg(test_theta,x2,y2,10)
print("Theta all ones and lambda =10")
print(f'Cost: {cost}')
print(f'Gradient(first five elements): {grad[:5].round(4)}')
options={'maxiter':100}
res=optimize.minimize(costFunctionReg,initial_theta,(x2,y2,lambda_),jac=True,method='TNC',options=options)
cost=res.fun
theta=res.x
print(f'Cost at optimised theta: {cost.round(3)}')
print(f'The optimised theta: {theta}')
p=predict(theta,x2)
print(f'Train Accuracy: {np.mean(p==y2)*100}')


