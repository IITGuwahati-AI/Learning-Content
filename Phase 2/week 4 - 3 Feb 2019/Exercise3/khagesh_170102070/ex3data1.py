import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.io import loadmat

def sigmoid(z):
    """
    Computes the sigmoid of z.
    """
    return 1.0 / (1.0 + np.exp(-z));

def lrCostFunction(theta, X, y, lambda_):
    #Initialize some useful values
    m = y.size
    
    # convert labels to ints if their type is bool
    if y.dtype == bool:
        y = y.astype(int)
    
    # You need to return the following variables correctly
    J = 0
    grad = np.zeros(theta.shape)
    
    # ====================== YOUR CODE HERE ======================
    h = sigmoid(np.dot(X,theta.T));
    temp = theta.copy();
    temp[0]=0;
    J = (-1) * (np.dot(y.T,np.log(h))+np.dot((1-y).T,np.log(1-h)))/m + (lambda_/(2*m))*np.sum(np.power(temp,2));
    grad = (1/m)*np.dot(X.T,h-y) + (lambda_/m)*temp;
        
    # =============================================================
    return J, grad

def oneVsAll(X, y, num_labels, lambda_):
    # Some useful variables
    m, n = X.shape
    
    # You need to return the following variables correctly 
    all_theta = np.zeros((num_labels, n + 1))

    # Add ones to the X data matrix
    X = np.concatenate([np.ones((m, 1)), X], axis=1)

    # ====================== YOUR CODE HERE ======================
    for c in range(num_labels):
        # Set Initial theta
        initial_theta = np.zeros(n + 1)
      
        # Set options for minimize
        options = {'maxiter': 50}
        res = optimize.minimize(lrCostFunction,initial_theta,(X, (y == c), lambda_),jac=True,method='TNC',options=options) 
        if c==0:
            all_theta = np.array([res.x]);
        else:
            all_theta = np.append(all_theta,[res.x],axis=0);
            
    # ============================================================
    return all_theta

def predictOneVsAll(all_theta, X):
    m = X.shape[0];
    num_labels = all_theta.shape[0]

    # You need to return the following variables correctly 
    p = np.zeros(m)

    # Add ones to the X data matrix
    X = np.concatenate([np.ones((m, 1)), X], axis=1)

    # ====================== YOUR CODE HERE ======================

    p = np.dot(X,all_theta.T);
    p = np.argmax(p, axis=1);
    
    # ============================================================
    return p

def image(X):
    for i in range(10):
        for j in range(10):
            if j==0:
                x = np.array(np.reshape(X[10*i+j], (-1, 20))).T;
            else:
                sample = np.array(np.reshape(X[10*i+j], (-1, 20))).T;
                x = np.hstack((x,sample));
        if i==0:
            graph = x.T;
        else:
            graph = np.hstack((graph,x.T));
    plt.figure(figsize=(5,5))
    plt.axis('off');
    arr = np.asarray(graph.T)
    plt.imshow(arr, cmap='gray')
    plt.show()

if __name__=='__main__':
    input_layer_size  = 400
    num_labels = 10
    data = loadmat(os.path.join('', 'ex3data1.mat'))
    X, y = data['X'], data['y'].ravel()
    y[y == 10] = 0

    m = y.size
    
    # Randomly select 100 data points to display
    rand_indices = np.random.choice(m, 100, replace=False)
    sel = X[rand_indices, :]
    image(sel);

#testing cost function and gradient descent    
    # test values for the parameters theta
    theta_t = np.array([-2, -1, 1, 2], dtype=float)

    # test values for the inputs
    X_t = np.concatenate([np.ones((5, 1)), np.arange(1, 16).reshape(5, 3, order='F')/10.0], axis=1)

    # test values for the labels
    y_t = np.array([1, 0, 1, 0, 1])

    # test value for the regularization parameter
    lambda_t = 3
    
    
    J, grad = lrCostFunction(theta_t, X_t, y_t, lambda_t)
#goto test examples
    
    
    lambda_ = 0.1
    all_theta = oneVsAll(X, y, num_labels, lambda_)
    
    pred = predictOneVsAll(all_theta, X)
    print('Training Set Accuracy: {:.2f}%'.format(np.mean(pred == y) * 100))

#predicted numbers
    print('----------Predicted Numbers----------');
    test = predictOneVsAll(all_theta, sel)
    for i in range(10):
        for j in range(10):
            print(test[i*10+j],'  ',end='');
        print();
        
#test examples  
    print('-----------------------')
    print('Cost         : {:.6f}'.format(J))
    print('Expected cost: 2.534819')
    print('-----------------------')
    print('Gradients:')
    print(' [{:.6f}, {:.6f}, {:.6f}, {:.6f}]'.format(*grad))
    print('Expected gradients:')
    print(' [0.146561, -0.548558, 0.724722, 1.398003]');
