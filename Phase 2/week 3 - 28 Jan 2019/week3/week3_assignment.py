import numpy as np
from matplotlib import pyplot as plt
import math
from scipy import optimize
import utils

data = np.array(np.loadtxt('ex2data1.txt',delimiter=','))
X,y = data[:,(0,1)],data[:,2]
y=np.reshape(y,(100,1))

 def plotData(X, y):
    for i in range(0,100):
        if (y[i]==1):
            plt.scatter(X[i,0],X[i,1],s=10,c='black',marker='x')
        else:
            plt.scatter(X[i,0],X[i,1],s=10,c='yellow',marker='o')

 plotData(X, y)
plt.xlabel("Exam 1 Score")
plt.ylabel("Exam 2 Score")
plt.show()

def sigmoid(z):
    if type(z)==int:
        s= (1/(1+math.exp((-1)*z)))
    else:
        m=np.size(z)
        s=np.exp(((-1)*z))
        for i in range(0,m):
            s[i]=(1/(1+s[i]))
    return s

m=np.shape(X)[0]
X = np.concatenate([np.ones((m, 1)), X], axis=1)
def CostFunction(theta, X, y):
    m=np.shape(X)[0]
    Cost=0
    for i in range(0,m):
        Cost=Cost-(y[i]*(math.log(sigmoid(np.dot(X[i,:],np.reshape(theta,(3,1))))))+(1-y[i])*(math.log(1-sigmoid(np.dot(X[i,:],np.reshape(theta,(3,1)))))))
    Cost=Cost/m

     grad = np.zeros((3,1))
    for i in range(0,3):
        a=0
        for j in range(0,m):
            a=a+((sigmoid(np.dot(X[j,:],np.reshape(theta,(3,1))))-y[j])*X[j,i])
        grad[i]=a/m
    return Cost,grad

initial_theta = np.zeros(3)

 cost, grad = CostFunction(initial_theta, X, y)

 print('Cost at initial theta (zeros): ',cost)
print('Expected cost (approx): 0.693\n')

 print('Gradient at initial theta (zeros):')
print(grad)
print('Expected gradients (approx):\n\t[-0.1000, -12.0092, -11.2628]\n')

 test_theta = np.array([[-24], [0.2], [0.2]])
cost, grad = CostFunction(test_theta, X, y)

 print('Cost at test theta: ',cost)
print('Expected cost (approx): 0.218\n')

 print('Gradient at test theta:')
print(grad)
print('Expected gradients (approx):\n\t[0.043, 2.566, 2.647]')


options= {'maxiter': 400}
res = optimize.minimize(CostFunction,initial_theta,(X, y),jac=True,method='TNC', options=options)
cost = res.fun
theta = res.x

 print('Cost at theta found by optimize.minimize: ',cost)
print('Expected cost (approx): 0.203\n')

 print('theta:')
print(theta)
print('Expected theta (approx):\n\t[-25.161, 0.206, 0.201]')
utils.plotDecisionBoundary(plotData, theta, X, y)
plt.show()

def predict(theta, X):
    m = X.shape[0]
    n=np.size(theta)
    p = np.zeros(m)
    for i in range(0,m):
        hypothesis=sigmoid(np.dot(X[i,:],np.reshape(theta,(n,1))))
        if hypothesis>= 0.5:
            p[i]=1
        elif(hypothesis< 0.5):
            p[i]=0
    return p

 prob = sigmoid(np.dot([1, 45, 85], np.reshape(theta,(3,1))))
print('For a student with scores 45 and 85,'
      'we predict an admission probability of ',prob)
print('Expected value: 0.775 +/- 0.002\n')

 p = predict(theta, X)
print('Train Accuracy: %'.format(np.mean(p == y) * 100))
print('Expected accuracy (approx): 89.00 %')


data = np.array(np.loadtxt('ex2data2.txt', delimiter=','))
X = data[:, :2]
y = data[:, 2]

plotData(X, y)
plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')
plt.show()
# Feature Mapping
X = utils.mapFeature(X[:, 0], X[:, 1])

def costFunctionReg(theta, X, y,lambda_):
    m = np.size(y)
    n=np.size(theta)
    J = 0
    grad = np.zeros(theta.shape)
    for i in range(0,m):
        J=J-(y[i]*(math.log(sigmoid(np.dot(X[i,:],np.reshape(theta,(28,1))))))+(1-y[i])*(math.log(1-sigmoid(np.dot(X[i,:],np.reshape(theta,(28,1)))))))
    J =(J/m)+((lambda_/(2*m))*(np.dot(theta,np.reshape(theta,(n,1)))-1))

     for i in range(0, 28):
        a = 0
        for j in range(0, m):
            a = a + ((sigmoid(np.dot(X[j, :], np.reshape(theta, (28, 1)))) - y[j]) * X[j, i])
        if i==0:
            grad[i] = a / m
        else:
            grad[i] = a / m+ (lambda_ / m) * theta[i]
    return J, grad

initial_theta = np.zeros(X.shape[1])
lambda_ = 1
cost, grad = costFunctionReg(initial_theta, X, y, lambda_)

 print('Cost at initial theta (zeros): {:.3f}',cost)
print('Expected cost (approx)       : 0.693\n')

 print('Gradient at initial theta (zeros) - first five values only:')
print(grad[:5])
print('Expected gradients (approx) - first five values only:')
print('\t[0.0085, 0.0188, 0.0001, 0.0503, 0.0115]\n')

 test_theta = np.ones(X.shape[1])
cost, grad = costFunctionReg(test_theta, X, y, 10)

 print('------------------------------\n')
print('Cost at test theta    : ',cost)
print('Expected cost (approx): 3.16\n')

 print('Gradient at initial theta (zeros) - first five values only:')
print(grad[:5])
print('Expected gradients (approx) - first five values only:')
print('\t[0.3460, 0.1614, 0.1948, 0.2269, 0.0922]')


initial_theta = np.zeros(X.shape[1])
lambda_ = 1
options= {'maxiter': 100}
res = optimize.minimize(costFunctionReg,initial_theta,(X, y, lambda_),jac=True, method='TNC',options=options)
cost = res.fun
theta = res.x

 utils.plotDecisionBoundary(plotData, theta, X, y)
plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')
plt.grid(False)
plt.title('lambda = %0.2f' % lambda_)
plt.show()

 p = predict(theta, X)

 print('Train Accuracy: %.1f %%' % (np.mean(p == y) * 100))
print('Expected accuracy (with lambda = 1): 83.1 % (approx)\n') 