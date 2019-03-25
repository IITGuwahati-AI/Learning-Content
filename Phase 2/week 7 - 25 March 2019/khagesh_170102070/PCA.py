import os
import numpy as np
import re
import utils
from PIL import Image
from matplotlib import pyplot
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl

from IPython.display import HTML, display, clear_output

try:
    pyplot.rcParams["animation.html"] = "jshtml"
except ValueError:
    pyplot.rcParams["animation.html"] = "html5"

from scipy import optimize
from scipy.io import loadmat


grader = utils.Grader()


def pca(X):
    # Useful values
    m, n = X.shape

    # You need to return the following variables correctly.
    U = np.zeros(n)
    S = np.zeros(n)

    # ====================== YOUR CODE HERE ======================

    sigma = (1/m)*np.dot(X.T,X)
    U , S , V = np.linalg.svd(sigma)
    
    # ============================================================
    return U, S

def projectData(X, U, K):
    # You need to return the following variables correctly.
    Z = np.zeros((X.shape[0], K))

    # ====================== YOUR CODE HERE ======================

    Z = np.dot(X,U[:,:K])
    
    # =============================================================
    return Z

def recoverData(Z, U, K):
    # You need to return the following variables correctly.
    X_rec = np.zeros((Z.shape[0], U.shape[0]))

    # ====================== YOUR CODE HERE ======================

    X_rec = np.dot(Z,U[:,:K].T)

    # =============================================================
    return X_rec

if __name__=='__main__':
    # Load the dataset into the variable X 
    data = loadmat(os.path.join('Data', 'ex7data1.mat'))
    X = data['X']
    
    #  Visualize the example dataset
    pyplot.plot(X[:, 0], X[:, 1], 'bo', ms=10, mec='k', mew=1)
    pyplot.axis([0.5, 6.5, 2, 8])
    pyplot.gca().set_aspect('equal')
    pyplot.grid(False)
    
    #---------------------------------------------------------------------------------
    
    #  Before running PCA, it is important to first normalize X
    X_norm, mu, sigma = utils.featureNormalize(X)
    
    #  Run PCA
    U, S = pca(X_norm)
    
    #  Draw the eigenvectors centered at mean of data. These lines show the
    #  directions of maximum variations in the dataset.
    fig, ax = pyplot.subplots()
    ax.plot(X[:, 0], X[:, 1], 'bo', ms=10, mec='k', mew=0.25)
    
    for i in range(2):
        ax.arrow(mu[0], mu[1], 1.5 * S[i]*U[0, i], 1.5 * S[i]*U[1, i],
                 head_width=0.25, head_length=0.2, fc='k', ec='k', lw=2, zorder=1000)
    
    ax.axis([0.5, 6.5, 2, 8])
    ax.set_aspect('equal')
    ax.grid(False)
    
    print('Top eigenvector: U[:, 0] = [{:.6f} {:.6f}]'.format(U[0, 0], U[1, 0]))
    print(' (you should expect to see [-0.707107 -0.707107])')
    
    #------------------------------------------------------------------------------------------
    
    #  Project the data onto K = 1 dimension
    K = 1
    Z = projectData(X_norm, U, K)
    print('Projection of the first example: {:.6f}'.format(Z[0, 0]))
    print('(this value should be about    : 1.481274)')
    
    #------------------------------------------------------------------------------------------
    
    X_rec  = recoverData(Z, U, K)
    print('Approximation of the first example: [{:.6f} {:.6f}]'.format(X_rec[0, 0], X_rec[0, 1]))
    print('       (this value should be about  [-1.047419 -1.047419])')
    
    #  Plot the normalized dataset (returned from featureNormalize)
    fig, ax = pyplot.subplots(figsize=(5, 5))
    ax.plot(X_norm[:, 0], X_norm[:, 1], 'bo', ms=8, mec='b', mew=0.5)
    ax.set_aspect('equal')
    ax.grid(False)
    pyplot.axis([-3, 2.75, -3, 2.75])
    
    # Draw lines connecting the projected points to the original points
    ax.plot(X_rec[:, 0], X_rec[:, 1], 'ro', mec='r', mew=2, mfc='none')
    for xnorm, xrec in zip(X_norm, X_rec):
        ax.plot([xnorm[0], xrec[0]], [xnorm[1], xrec[1]], '--k', lw=1)
    pyplot.show()
        
    #------------------------------------------------------------------------------------------
    
    #  Load Face dataset
    data = loadmat(os.path.join('Data', 'ex7faces.mat'))
    X = data['X']
    
    #  Display the first 100 faces in the dataset
    utils.displayData(X[:100, :], figsize=(8, 8))

    #  normalize X by subtracting the mean value from each feature
    X_norm, mu, sigma = utils.featureNormalize(X)
    
    #  Run PCA
    U, S = pca(X_norm)
    
    #  Visualize the top 36 eigenvectors found
    utils.displayData(U[:, :36].T, figsize=(8, 8))
    
    #  Project images to the eigen space using the top k eigenvectors 
    #  If you are applying a machine learning algorithm 
    K = 100
    Z = projectData(X_norm, U, K)
    
    print('The projected data Z has a shape of: ', Z.shape)

    #  Project images to the eigen space using the top K eigen vectors and 
    #  visualize only using those K dimensions
    #  Compare to the original input, which is also displayed
    K = 100
    X_rec  = recoverData(Z, U, K)
    
    # Display normalized data
    utils.displayData(X_norm[:100, :], figsize=(6, 6))
    pyplot.gcf().suptitle('Original faces')
    
    # Display reconstructed data from only k eigenfaces
    utils.displayData(X_rec[:100, :], figsize=(6, 6))
    pyplot.gcf().suptitle('Recovered faces')
    pass



    #------------------------------Optional---------------------------------------
    
    















