import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize
def plotData(X, y):
    pos = X[np.where(y==1)]
    neg = X[np.where(y==0)]
    fig, ax = plt.subplots()
    ax.plot(pos[:,0],pos[:,1],"k+",neg[:,0],neg[:,1],"yo")
    return (fig, ax)
def costFunction(theta,X,y):
    
    m = len(y) 
    J =(np.sum(-y*np.log(sigmoid(np.dot(X,theta)))-
       (1-y)*(np.log(1-sigmoid(np.dot(X,theta)))))/m)
    grad = (np.sum((sigmoid(np.dot(X,theta))-y)[:,None]*X,axis=0)/m)
    return (J, grad)
def sigmoid(z):
     return 1.0/(1 +  np.e**(-z))

def predict(theta,X):
    return np.where(np.dot(X,theta) > 5.,1,0)
def mapFeatureVector(X1,X2):
    degree = 6
    output_feature_vec = np.ones(len(X1))[:,None]
    for i in range(1,7):
        for j in range(i+1):
            new_feature = np.array(X1**(i-j)*X2**j)[:,None]
            output_feature_vec = np.hstack((output_feature_vec,new_feature))
   
    return output_feature_vec
def costFunctionReg(theta,X,y,reg_param):
    m = len(y)  
    grad = (np.sum((sigmoid(np.dot(X,theta))-y)[:,None]*X,axis=0)/m)
    grad_reg = grad + (reg_param/m)*theta
    grad_reg[0] = grad[0] 
    J =(np.sum(-y*np.log(sigmoid(np.dot(X,theta)))-(1-y)*(np.log(1-sigmoid(np.dot(X,theta))))))/m + (reg_param/m)*np.sum(theta**2) 
    return J
def plotDecisionBoundary(theta,X,y):
    fig, ax = plotData(X[:,1:],y)
    u = np.linspace(-1,1.5,50)
    v = np.linspace(-1,1.5,50)
    z = np.zeros((len(u),len(v)))
    for i in range(len(u)):
        for j in range(len(v)):
            z[i][j] = np.dot(mapFeatureVector(np.array([u[i]]),
		      np.array([v[j]])),theta)
    ax.contour(u,v,z,levels=[0])
    return (fig,ax)
data = pd.read_csv('Data/ex2data2.txt', names=['x1','x2','y'])
X = np.asarray(data[["x1","x2"]])
y = np.asarray(data["y"])
fig, ax = plotData(X, y)
fig.show()
X = mapFeatureVector(X[:,0],X[:,1])
initial_theta = np.zeros(len(X[0,:]))
reg_param = 1.0
res = minimize(costFunctionReg,
	       initial_theta,
	       args=(X,y,reg_param),
	       tol=1e-6,
	       options={'maxiter':400,
			'disp':True})
theta = res.x
fig, ax = plotDecisionBoundary(theta,X,y)
ax.set_title('Perfect')
fig.show()
reg_param = 15.0
res = minimize(costFunctionReg,
	       initial_theta,
	       args=(X,y,reg_param),
	       tol=1e-6,
	       options={'maxiter':400,
			'disp':True})
theta = res.x
fig, ax = plotDecisionBoundary(theta,X,y)
ax.set_title('Underfitting ')
fig.show()
reg_param = 0
res = minimize(costFunctionReg,
	       initial_theta,
	       args=(X,y,reg_param),
	       tol=1e-6,
	       options={'maxiter':400,
			'disp':True})
theta = res.x
fig, ax = plotDecisionBoundary(theta,X,y)
ax.set_title('Overfitting')
fig.show()