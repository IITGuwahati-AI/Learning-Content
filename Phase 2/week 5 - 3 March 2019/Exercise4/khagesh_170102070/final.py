import os
import numpy as np
from matplotlib import pyplot
from scipy import optimize
from scipy.io import loadmat
import utils
from math import sqrt;
grader = utils.Grader()

def nnCostFunction(nn_params,input_layer_size, hidden_layer_size,num_labels, X, y, lambda_=0.0):
    Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],  (hidden_layer_size, (input_layer_size + 1)))

    Theta2 = np.reshape(nn_params[(hidden_layer_size * (input_layer_size + 1)):],  (num_labels, (hidden_layer_size + 1)))

    # Setup some useful variables
    m = y.size
         
    # You need to return the following variables correctly 
    J = 0
    Theta1_grad = np.zeros(Theta1.shape)
    Theta2_grad = np.zeros(Theta2.shape)

    # ====================== YOUR CODE HERE ======================
    X = np.append( np.ones((1,m)) , X.T ,axis=0)
    h2 = utils.sigmoid(np.dot(Theta1,X))
    h2 = np.append( np.ones((1,m)) , h2 , axis=0 )
    h3 = utils.sigmoid( np.dot(Theta2,h2) );
    
    Y = np.zeros((num_labels,m));
    for c in range(num_labels):
        Y[c,y==c] = 1;
    J = np.sum( -np.multiply(Y,np.log(h3)) - np.multiply(1-Y,np.log(1-h3)) )/m + (lambda_/(2*m))*(np.sum( np.power( Theta1[:,1:],2 ) ) + np.sum( np.power( Theta2[:,1:],2 ) ) );
    
    small_delta3 = h3 - Y;
    small_delta2 = np.dot(Theta2.T , small_delta3) * h2 * (1-h2)
    
    theta2 = Theta2.copy();
    theta1 = Theta1.copy();
    theta2[:,0] = 0;
    theta1[:,0] = 0;
    Theta2_grad = (1/m)*np.dot( small_delta3,h2.T ) + (lambda_/m)*theta2;
    Theta1_grad = (1/m)*np.dot( small_delta2[1:,:], X.T ) + (lambda_/m)*theta1;
    
    # ================================================================
    # Unroll gradients
    # grad = np.concatenate([Theta1_grad.ravel(order=order), Theta2_grad.ravel(order=order)])
    grad = np.concatenate([Theta1_grad.ravel(), Theta2_grad.ravel()])
    return J, grad

def sigmoidGradient(z):
    g = np.zeros(z.shape)

    # ====================== YOUR CODE HERE ======================

    g = utils.sigmoid(z) * (1-utils.sigmoid(z))

    # =============================================================
    return g

def randInitializeWeights(L_in, L_out, epsilon_init=0.12):
    # You need to return the following variables correctly 
    W = np.zeros((L_out, 1 + L_in))

    # ====================== YOUR CODE HERE ======================

    W = np.random.randint(-50000,50000,(L_out,1+L_in)) *(epsilon_init/100000)
    #print(W)

    # ============================================================
    return W

def image(X,figsize=(5,5)):
    m,n = X.shape
    m = int(sqrt(m))
    n = int(sqrt(n))
    for i in range(m):
        for j in range(m):
            if j==0:
                x = np.array(np.reshape(X[m*i+j], (-1, n))).T;
            else:
                sample = np.array(np.reshape(X[m*i+j], (-1, n))).T;
                x = np.hstack((x,sample));
        if i==0:
            graph = x.T;
        else:
            graph = np.hstack((graph,x.T));
    pyplot.figure(figsize=figsize)
    pyplot.axis('off');
    arr = np.asarray(graph.T)
    pyplot.imshow(arr, cmap='gray')
    pyplot.show()

if __name__=='__main__':
    data = loadmat(os.path.join('', 'ex4data1.mat'))
    X, y = data['X'], data['y'].ravel()
    y[y == 10] = 0
    m = y.size
    #print(X.shape,y.shape);
    rand_indices = np.random.choice(m, 100, replace=False)
    sel = X[rand_indices, :]
    #utils.displayData(sel)
    image(sel);
    
    # Setup the parameters you will use for this exercise
    input_layer_size  = 400  # 20x20 Input Images of Digits
    hidden_layer_size = 25   # 25 hidden units
    num_labels = 10          # 10 labels, from 0 to 9

    # Load the weights into variables Theta1 and Theta2
    weights = loadmat(os.path.join('', 'ex4weights.mat'))

    # Theta1 has size 25 x 401
    # Theta2 has size 10 x 26
    Theta1, Theta2 = weights['Theta1'], weights['Theta2']

    # swap first and last columns of Theta2, due to legacy from MATLAB indexing, 
    # since the weight file ex3weights.mat was saved based on MATLAB indexing
    Theta2 = np.roll(Theta2, 1, axis=0)

    # Unroll parameters 
    nn_params = np.concatenate([Theta1.ravel(), Theta2.ravel()])
    
    
    lambda_ = 0
    J, _ = nnCostFunction(nn_params, input_layer_size, hidden_layer_size,
                   num_labels, X, y, lambda_)
    print('Cost at parameters (loaded from ex4weights): %.6f ' % J)
    print('The cost should be about                   : 0.287629.')
    #utils.checkNNGradients(nnCostFunction, lambda_=0)
    
    #grader = utils.Grader()
    #grader[1] = nnCostFunction
    #grader.grade()
    ###########################################################################
    z = np.array([-1, -0.5, 0, 0.5, 1])
    g = sigmoidGradient(z)
    print('Sigmoid gradient evaluated at [-1 -0.5 0 0.5 1]:\n  ')
    print(g)
    
    #grader[3] = sigmoidGradient
    #grader.grade()
    
    
    print('Initializing Neural Network Parameters ...')

    initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
    initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels)

    # Unroll parameters
    initial_nn_params = np.concatenate([initial_Theta1.ravel(), initial_Theta2.ravel()], axis=0)
    
    
    #utils.checkNNGradients(nnCostFunction)
    
    #  Check gradients by running checkNNGradients
    lambda_ = 3
    utils.checkNNGradients(nnCostFunction, lambda_)

    # Also output the costFunction debugging values
    debug_J, _  = nnCostFunction(nn_params, input_layer_size,hidden_layer_size, num_labels, X, y, lambda_)

    print('\n\nCost at (fixed) debugging parameters (w/ lambda = %f): %f ' % (lambda_, debug_J))
    print('(for lambda = 3, this value should be about 0.576051)')
    
    ##########################################################################
    
    #  After you have completed the assignment, change the maxiter to a larger
    #  value to see how more training helps.
    options= {'maxiter': 100}

    #  You should also try different values of lambda
    lambda_ = 1

    # Create "short hand" for the cost function to be minimized
    costFunction = lambda p: nnCostFunction(p, input_layer_size,hidden_layer_size,num_labels, X, y, lambda_)

    # Now, costFunction is a function that takes in only one argument
    # (the neural network parameters)
    res = optimize.minimize(costFunction, initial_nn_params, jac=True, method='TNC', options=options)

    # get the solution of the optimization
    nn_params = res.x
        
    # Obtain Theta1 and Theta2 back from nn_params
    Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],(hidden_layer_size, (input_layer_size + 1)))

    Theta2 = np.reshape(nn_params[(hidden_layer_size * (input_layer_size + 1)):],(num_labels, (hidden_layer_size + 1)))
    
    pred = utils.predict(Theta1, Theta2, X)
    print('Training Set Accuracy: %f' % (np.mean(pred == y) * 100))
    
    utils.displayData(Theta1[:, 1:])
    test_data = Theta1[:, 1:]
    #image(test_data)
    m,n = test_data.shape
    m = int(sqrt(m))
    n = int(sqrt(n))
    for i in range(m):
        for j in range(m):
            #print(test_data[i][j],end='')
            pass
        print()




















