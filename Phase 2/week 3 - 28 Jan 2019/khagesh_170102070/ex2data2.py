import numpy as np;
import matplotlib.pyplot as plt;
from matplotlib.patches import Patch;
from matplotlib.lines import Line2D;
from sklearn.model_selection import train_test_split;
import pandas as pd;

def logistic_expression(X,theta):
    h = 1/(1+np.exp((-1)*np.dot(X,theta)));
    return h;

def cost_function(X,Y,theta,h):
    beta = np.copy(theta);
    beta[0][0] = 0.0;
    J = (-1) * (np.dot(Y.T,np.log(h))+np.dot((1-Y).T,np.log(1-h))) + (np.power(beta,2))/2;
    return np.mean(J);

def gradient(X,Y,h,theta):
    beta = np.copy(theta);
    beta[0][0] = 0.0;
    grad = np.dot(X.T,h-Y)+beta;
    return grad;

def gradient_descent(X,Y,theta,m,alpha = 0.01):
    h = logistic_expression(X,theta);
    cost = cost_function(X,Y,theta,h);
    change = 1;
    iterations = 1;
    while(change > 0.0001):
        old_cost = cost;
        theta = theta - (alpha * gradient(X,Y,h,theta))/m;
        h = logistic_expression(X,theta);
        cost = cost_function(X,Y,theta,h);
        change = old_cost-cost;
        iterations+=1;
    return theta, iterations;

def reshape(x,degree):
    X = np.ones(x.shape[0])[:,np.newaxis]
    for i in range(1, degree+1):
        for j in range(i+1):
            X = np.hstack((X, np.multiply(np.power(x[:,0], i-j), np.power(x[:,1], j))[:,np.newaxis]))
    return X

def equation(theta,degree):
    x = np.linspace(-1, 1.5, 100)
    y = np.linspace(-1, 1.5, 100)
    X, Y = np.meshgrid(x, y)
    Z = theta[0][0];
    count=1;
    for i in range(1, degree+1):
        for j in range(i+1):
            Z += theta[count][0]*np.multiply(np.power(X,i-j), np.power(Y,j));
            count+=1;
    return x,y,Z;

def plot_reg(X, Y, theta,m,degree): 
    y = [None]*m;
    for i in range(0,m):
        if Y[i][0] == 0:
            y[i] = 'b';
        else:
            y[i] = 'r';
    fig, ax = plt.subplots();
    plt.scatter(X[:,1],X[:,2],c=y,s=20);
      
    # plotting decision boundary 
    x,y,z = equation(theta,degree);
    plt.contour(x,y,z,0,colors = 'black',linestyles='solid',linewidths=1.0 );
  
    plt.xlabel('Microchip test1') 
    plt.ylabel('Microchip test2') 
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='Passed', markerfacecolor='r', markersize=10),
                   Line2D([0], [0], marker='o', color='w', label='Failed', markerfacecolor='b', markersize=10)];

    # Create the figure
    
    ax.legend(handles=legend_elements, loc='upper right')
    #plt.legend() 
    plt.show()
    
def probability(theta,X_test,Y_test,degree):
    count=1;
    m = X_test.shape[0];
    Z=theta[0][0]*np.array([X_test[:,0]]);
    for i in range(1, degree+1):
        for j in range(i+1):
            Z += theta[count][0]*np.multiply(np.power(np.array([X_test[:,1]]),i-j), np.power(np.array([X_test[:,2]]),j));
            count+=1;
    status = list();
    for i in range(m):
        if Z[0][i]<0:
            #print('-> Passed tests');
            status.append('Passes');
        else:
            #print('-> Failed tests');
            status.append('Failed');
    X_test = X_test.T;
    dict = {'Test1': X_test[1].tolist(),
        'Test2': X_test[2].tolist(),
        'Status':status};
    df = pd.DataFrame(dict);
    df.fillna(0)
    print(df);
            
        
if __name__=='__main__':
    data = np.loadtxt('ex2data2.txt',delimiter=',');
    degree=6
    X = reshape(data,degree);
    Y = data[:,2:3];
    m,n = X.shape;
    theta = np.array([np.zeros(np.size(X[0]))]).T;
    theta,iterations = gradient_descent(X,Y,theta,m);
    
    plot_reg(np.hstack((np.array([np.ones(m)]).T,data)),Y,theta,m,degree);
    
    print('theta:',theta.T);
    print('No of iterations',iterations);
    print('-------------------------------------------------------------------');
    
    X_train, X_test, Y_train, Y_test = train_test_split( X,Y,test_size=0.05);
    probability(theta,X_test,Y_test,degree);
    
    
    
    
    
    
    
    
    
    
