# used for manipulating directory paths
import os

# Scientific and vector computation for python
import numpy as np

# Plotting library
from matplotlib import pyplot

# Optimization module in scipy
from scipy import optimize

# will be used to load MATLAB mat datafile format
from scipy.io import loadmat

# library written for this exercise providing additional functions for assignment submission, and others
import utils

def sigmoid(mat):
	return 1/(1+np.exp(-mat))

def nnCostFunction(nn_params,
                   input_layer_size,
                   hidden_layer_size,
                   num_labels,
                   X, y, lambda_=0.0):
    """
    Implements the neural network cost function and gradient for a two layer neural 
    network which performs classification. 
    
    Parameters
    ----------
    nn_params : array_like
        The parameters for the neural network which are "unrolled" into 
        a vector. This needs to be converted back into the weight matrices Theta1
        and Theta2.
    
    input_layer_size : int
        Number of features for the input layer. 
    
    hidden_layer_size : int
        Number of hidden units in the second layer.
    
    num_labels : int
        Total number of labels, or equivalently number of units in output layer. 
    
    X : array_like
        Input dataset. A matrix of shape (m x input_layer_size).
    
    y : array_like
        Dataset labels. A vector of shape (m,).
    
    lambda_ : float, optional
        Regularization parameter.
 
    Returns
    -------
    J : float
        The computed value for the cost function at the current weight values.
    
    grad : array_like
        An "unrolled" vector of the partial derivatives of the concatenatation of
        neural network weights Theta1 and Theta2.
    
    Instructions
    ------------
    You should complete the code by working through the following parts.
    
    - Part 1: Feedforward the neural network and return the cost in the 
              variable J. After implementing Part 1, you can verify that your
              cost function computation is correct by verifying the cost
              computed in the following cell.
    
    - Part 2: Implement the backpropagation algorithm to compute the gradients
              Theta1_grad and Theta2_grad. You should return the partial derivatives of
              the cost function with respect to Theta1 and Theta2 in Theta1_grad and
              Theta2_grad, respectively. After implementing Part 2, you can check
              that your implementation is correct by running checkNNGradients provided
              in the utils.py module.
    
              Note: The vector y passed into the function is a vector of labels
                    containing values from 0..K-1. You need to map this vector into a 
                    binary vector of 1's and 0's to be used with the neural network
                    cost function.
     
              Hint: We recommend implementing backpropagation using a for-loop
                    over the training examples if you are implementing it for the 
                    first time.
    
    - Part 3: Implement regularization with the cost function and gradients.
    
              Hint: You can implement this around the code for
                    backpropagation. That is, you can compute the gradients for
                    the regularization separately and then add them to Theta1_grad
                    and Theta2_grad from Part 2.
    
    Note 
    ----
    We have provided an implementation for the sigmoid function in the file 
    `utils.py` accompanying this assignment.
    """
    # Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
    # for our 2 layer neural network
    Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],
                        (hidden_layer_size, (input_layer_size + 1)))

    Theta2 = np.reshape(nn_params[(hidden_layer_size * (input_layer_size + 1)):],
                        (num_labels, (hidden_layer_size + 1)))

    # Setup some useful variables
    m = y.size
    # You need to return the following variables correctly 
    J = 0
    # Theta1_grad = np.zeros(Theta1.shape)
    # Theta2_grad = np.zeros(Theta2.shape)

    # ====================== YOUR CODE HERE ======================
    X = np.append(np.ones((X.shape[0],1),dtype = float),X,axis = 1)
    y_mat = np.zeros((m,num_labels))
    y_mat[np.arange(m),y] = 1
    h = sigmoid(np.matmul(X,Theta1.T))
    h = np.append(np.ones((h.shape[0],1),dtype = float),h,axis = 1)
    h = sigmoid(np.matmul(h,Theta2.T))
    for i in range(m):
    	a = np.matmul(y_mat[i],np.log(h[i]))
    	b = np.matmul(1-y_mat[i],np.log(1-h[i]))
    	J += -1*(a+b)/m
    J += (lambda_/(2*m))*(np.sum(np.square(Theta1))+np.sum(np.square(Theta2)))
    J -= (lambda_/(2*m))*(np.matmul(Theta1[:,0],Theta1[:,0])+np.matmul(Theta2[:,0],Theta2[:,0]))
    #gradient
    delta_1 = np.zeros(Theta1.shape)
    delta_2 = np.zeros(Theta2.shape)
    sdel_2 = None
    sdel_3 = h-y_mat
    for i in range(m):
        z_2 = np.matmul(Theta1,X[i])
        a_2 = sigmoid(z_2)
        g_dash_z_2 = sigmoidGradient(z_2)
        a_2 = np.append(np.ones(1),a_2)
        g_dash_z_2 = np.append(np.ones(1),g_dash_z_2)
        sdel_2 = np.matmul(Theta2.T,sdel_3[i])
        sdel_2 = sdel_2*g_dash_z_2
        sdel_2 = sdel_2[1:]
        delta_2 += np.matmul(sdel_3[i].reshape(len(sdel_3[i]),1),a_2.reshape(len(a_2),1).T)
        delta_1 += np.matmul(sdel_2.reshape(len(sdel_2),1),X[i].reshape(len(X[i]),1).T)
    delta_1 /= m
    delta_2 /= m
    delta_1[:,1:] += (lambda_/m)*Theta1[:,1:]
    delta_2[:,1:] += (lambda_/m)*Theta2[:,1:]
    # ================================================================
    # Unroll gradients
    # grad = np.concatenate([Theta1_grad.ravel(order=order), Theta2_grad.ravel(order=order)])
    grad = np.concatenate([delta_1.ravel(), delta_2.ravel()])
    return J, grad

def sigmoidGradient(z):
    """
    Computes the gradient of the sigmoid function evaluated at z. 
    This should work regardless if z is a matrix or a vector. 
    In particular, if z is a vector or matrix, you should return
    the gradient for each element.
    
    Parameters
    ----------
    z : array_like
        A vector or matrix as input to the sigmoid function. 
    
    Returns
    --------
    g : array_like
        Gradient of the sigmoid function. Has the same shape as z. 
    
    Instructions
    ------------
    Compute the gradient of the sigmoid function evaluated at
    each value of z (z can be a matrix, vector or scalar).
    
    Note
    ----
    We have provided an implementation of the sigmoid function 
    in `utils.py` file accompanying this assignment.
    """

    g = np.zeros(z.shape)

    # ====================== YOUR CODE HERE ======================
    g = sigmoid(z)
    # =============================================================
    return g-g**2

def randInitializeWeights(L_in, L_out, epsilon_init=0.12):
    """
    Randomly initialize the weights of a layer in a neural network.
    
    Parameters
    ----------
    L_in : int
        Number of incomming connections.
    
    L_out : int
        Number of outgoing connections. 
    
    epsilon_init : float, optional
        Range of values which the weight can take from a uniform 
        distribution.
    
    Returns
    -------
    W : array_like
        The weight initialiatized to random values.  Note that W should
        be set to a matrix of size(L_out, 1 + L_in) as
        the first column of W handles the "bias" terms.
        
    Instructions
    ------------
    Initialize W randomly so that we break the symmetry while training
    the neural network. Note that the first column of W corresponds 
    to the parameters for the bias unit.
    """

    # You need to return the following variables correctly 
    W = np.zeros((L_out, 1 + L_in))

    # ====================== YOUR CODE HERE ======================
    # Randomly initialize the weights to small values
    W = np.random.rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init
    # ============================================================
    return W

