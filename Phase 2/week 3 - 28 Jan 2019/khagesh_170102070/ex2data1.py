import numpy as np;
import matplotlib.pyplot as plt;
from sklearn.model_selection import train_test_split;

def normalize(X): 
    mins = np.min(X, axis = 0) 
    maxs = np.max(X, axis = 0) 
    rng = maxs - mins;
    norm_X = 1 - ((maxs - X)/rng)
    return norm_X,maxs,rng;

def logistic_expression(X,theta):
    h = 1/(1+np.exp((-1)*np.dot(X,theta)));
    return h;

def cost_function(X,Y,theta,h):
    J = (-1) * (np.dot(Y.T,np.log(h))+np.dot((1-Y).T,np.log(1-h)));
    return np.mean(J);

def gradient(X,Y,h):
    grad = np.dot(X.T,h-Y);
    return grad;

def gradient_descent(X,Y,theta,m,alpha = 0.01):
    h = logistic_expression(X,theta);
    cost = cost_function(X,Y,theta,h);
    change = 1;
    iterations = 1;
    while(change > 0.000001):
        old_cost = cost;
        theta = theta - (alpha * gradient(X,Y,h))/1;
        h = logistic_expression(X,theta);
        cost = cost_function(X,Y,theta,h);
        change = old_cost-cost;
        iterations+=1;
    return theta, iterations;

def plot_reg(X, Y, theta,m): 
    y = [None]*m;
    for i in range(0,m):
        if Y[i][0] == 0:
            y[i] = 'b';
        else:
            y[i] = 'r';
    
    X = X.T;
    plt.scatter(X[1],X[2],c=y,s=20);
      
    # plotting decision boundary 
    x1 = np.arange(30, 100, 1) 
    x2 = -(theta[0][0] + theta[1][0]*x1)/theta[2][0] 
    plt.plot(x1, x2, c='g',linewidth=1.0) 
  
    plt.xlabel('x1') 
    plt.ylabel('x2') 
    #plt.legend() 
    plt.show()
    
def original(xmax,xrng,theta):
    beta = np.array([[0],[0],[0]],dtype='float64');
    beta[0][0] = np.sum(theta)-(theta[1][0]*xmax[0])/xrng[0]-(theta[2][0]*xmax[1])/xrng[1];
    beta[1][0] = theta[1][0]/xrng[0];
    beta[2][0] = theta[2][0]/xrng[1];
    return beta
    
def probability(theta,X_test,Y_test,xmax,xrng):
    X_local = 1 - ((xmax - X_test)/xrng);
    pred_prob = logistic_expression(np.hstack((np.array([np.ones(X_local.shape[0])]).T,X_local)),theta) 
    pred_value = np.where(pred_prob >= .5, 1, 0)
    m=np.size(pred_value)
    for i in range(m):
        print('Applicant with scores {:2f} : {:2f} has probability of {:2f}'.format(X_test[i][0],X_test[i][1],pred_prob[i][0]))
        
def predict(xmax,xrng,theta):
    try:
        x1,x2 = input('Enter scores in a line with space in btwn:').split(); 
    except ValueError:
        print('Error: Please input two numbers in a line with space');
        predict(xmax,xrng,theta)
        return ;
    except:
        return ;
    X = np.array([float(x1),float(x2)]);
    X = 1 - ((xmax - X)/xrng);
    X = np.hstack((np.array([1]).T,X));
    print('->Probability: {:2f}'.format(logistic_expression(X,theta)[0]));

if __name__=='__main__':
    data = np.loadtxt('../courseera/machine-learning-ex2/ex2/ex2data1.txt',delimiter=',');
    m = data.shape[0];
    X,xmax,xrng = normalize(data[:, :-1]) 
    X = np.hstack((np.array([np.ones(m)]).T,X));
    Y = data[:,2:3]
    theta = np.array([np.zeros(np.size(X[0]))]).T;
    theta,iterations = gradient_descent(X,Y,theta,m);
    
    plot_reg(np.hstack((np.array([np.ones(m)]).T,data)),Y,original(xmax,xrng,theta),m);
    print('theta:',theta.T);
    print('no of iterations',iterations);
    print('-------------------------------------------------------------------');
    
    X_train, X_test, Y_train, Y_test = train_test_split( data[:,0:2],Y,test_size=0.05);
    #random test cases
    probability(theta,X_test,Y_test,xmax,xrng);
    print('-------------------------------------------------------------------');
    #input test case
    predict(xmax,xrng,theta)
    
    
    
    
    
