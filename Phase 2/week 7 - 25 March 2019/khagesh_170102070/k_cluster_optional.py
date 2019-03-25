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

def findClosestCentroids(X, centroids):
    # Set K
    K = centroids.shape[0]

    # You need to return the following variables correctly.
    idx = np.zeros(X.shape[0], dtype=int)

    # ====================== YOUR CODE HERE ======================

    calc_centroid_matrix = np.zeros((X.shape[0],K))
    for i in range(K):
        calc_centroid_matrix[:,i] = np.sum(np.power(X-centroids[i],2),axis=1)
    calc_centroid_matrix = np.argmin(calc_centroid_matrix,axis=1)
    
    idx = calc_centroid_matrix
    
    # =============================================================
    return idx

def computeCentroids(X, idx, K):
    # Useful variables
    m, n = X.shape
    # You need to return the following variables correctly.
    centroids = np.zeros((K, n))


    # ====================== YOUR CODE HERE ======================

    for i in range(K):
        centroids[i,:] = np.mean(X[idx==i],axis=0)

    #for i in range(K):
    #    count=0
    #    for j in range(m):
    #        if idx[j]==i:
    #            count+=1
    #            centroids[i]+=X[j]
    #    centroids[i]/=count
    #    print(count)
    
    # =============================================================
    return centroids

def kMeansInitCentroids(X, K):
    m, n = X.shape
    
    # You should return this values correctly
    centroids = np.zeros((K, n))

    # ====================== YOUR CODE HERE ======================

    # Randomly reorder the indices of examples
    randidx = np.random.permutation(X.shape[0])
    # Take the first K examples as centroids
    centroids = X[randidx[:K], :]
    
    # =============================================================
    return centroids

if __name__=='__main__':
    # this allows to have interactive plot to rotate the 3-D plot
    # The double identical statement is on purpose
    # see: https://stackoverflow.com/questions/43545050/using-matplotlib-notebook-after-matplotlib-inline-in-jupyter-notebook-doesnt
    
    from matplotlib import pyplot
    
    
    A = mpl.image.imread(os.path.join('Data', 'bird_small.png'))
    A /= 255
    X = A.reshape(-1, 3)
    
    # perform the K-means clustering again here
    K = 16
    max_iters = 10
    initial_centroids = kMeansInitCentroids(X, K)
    centroids, idx = utils.runkMeans(X, initial_centroids,
                                     findClosestCentroids,
                                     computeCentroids, max_iters)
    
    #  Sample 1000 random indexes (since working with all the data is
    #  too expensive. If you have a fast computer, you may increase this.
    sel = np.random.choice(X.shape[0], size=1000)
    
    fig = pyplot.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(X[sel, 0], X[sel, 1], X[sel, 2], cmap='rainbow', c=idx[sel], s=8**2)
    ax.set_title('Pixel dataset plotted in 3D.\nColor shows centroid memberships')
    pass

    #-------------------------------------------------------------------------------------
    
    # Subtract the mean to use PCA
    X_norm, mu, sigma = utils.featureNormalize(X)
    
    # PCA and project the data to 2D
    U, S = pca(X_norm)
    Z = projectData(X_norm, U, 2)
    
    # Reset matplotlib to non-interactive
    
    fig = pyplot.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    
    ax.scatter(Z[sel, 0], Z[sel, 1], cmap='rainbow', c=idx[sel], s=64)
    ax.set_title('Pixel dataset plotted in 2D, using PCA for dimensionality reduction')
    ax.grid(False)
    pass
























