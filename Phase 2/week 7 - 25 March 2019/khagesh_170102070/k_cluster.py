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

def runkMeans_non_utils(X, centroids, findClosestCentroids, computeCentroids,
              max_iters=10, plot_progress=False):
    K = centroids.shape[0]
    idx = None
    idx_history = []
    centroid_history = []

    for i in range(max_iters):
        idx = findClosestCentroids(X, centroids)

        if plot_progress:
            idx_history.append(idx)
            centroid_history.append(centroids)

        centroids = computeCentroids(X, idx, K)
    utils.plotProgresskMeans(i, X, centroid_history, idx_history)
    
    anim='<Figure size 432x288 with 0 Axes>'
    return centroids, idx, anim

if __name__=='__main__':
    # Load an example dataset that we will be using
    data = loadmat(os.path.join('Data', 'ex7data2.mat'))
    X = data['X']
    
    # Select an initial set of centroids
    K = 3   # 3 Centroids
    initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])
    
    # Find the closest centroids for the examples using the initial_centroids
    idx = findClosestCentroids(X, initial_centroids)
    
    print('Closest centroids for the first 3 examples:')
    print(idx[:3])
    print('(the closest centroids should be 0, 2, 1 respectively)')
    
    #--------------------------------------------------------------------------------------------

    # Compute means based on the closest centroids found in the previous part.
    centroids = computeCentroids(X, idx, K)
    
    print('Centroids computed after initial finding of closest centroids:')
    print(centroids)
    print('\nThe centroids should be')
    print('   [ 2.428301 3.157924 ]')
    print('   [ 5.813503 2.633656 ]')
    print('   [ 7.119387 3.616684 ]')

    #--------------------------------------------------------------------------------------------

    # Load an example dataset
    data = loadmat(os.path.join('Data', 'ex7data2.mat'))
    
    # Settings for running K-Means
    K = 3
    max_iters = 10
    
    # For consistency, here we set centroids to specific values
    # but in practice you want to generate them automatically, such as by
    # settings them to be random examples (as can be seen in
    # kMeansInitCentroids).
    initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])
    
    
    # Run K-Means algorithm. The 'true' at the end tells our function to plot
    # the progress of K-Means
    #centroids, idx, anim = utils.runkMeans(X, initial_centroids,findClosestCentroids, computeCentroids, max_iters, True)
    #anim
    centroids, idx, anim = runkMeans_non_utils(X, initial_centroids,findClosestCentroids, computeCentroids, max_iters, True)

    

    # ======= Experiment with these parameters ================
    # You should try different values for those parameters
    K = 16
    max_iters = 10
    
    # Load an image of a bird
    # Change the file name and path to experiment with your own images
    A = mpl.image.imread(os.path.join('Data', 'bird_small.png'))
    print(A.shape)
    # ==========================================================
    
    # Divide by 255 so that all values are in the range 0 - 1
    A /= 255
    
    # Reshape the image into an Nx3 matrix where N = number of pixels.
    # Each row will contain the Red, Green and Blue pixel values
    # This gives us our dataset matrix X that we will use K-Means on.
    X = A.reshape(-1, 3)
    print(X.shape)
    
    # When using K-Means, it is important to randomly initialize centroids
    # You should complete the code in kMeansInitCentroids above before proceeding
    initial_centroids = kMeansInitCentroids( X, K)
    
    # Run K-Means
    centroids, idx = utils.runkMeans(X, initial_centroids,
                                     findClosestCentroids,
                                     computeCentroids,
                                     max_iters)
    print(centroids.shape)
    # We can now recover the image from the indices (idx) by mapping each pixel
    # (specified by its index in idx) to the centroid value
    # Reshape the recovered image into proper dimensions
    X_recovered = centroids[idx, :].reshape(A.shape)
    print(X_recovered.shape)
    
    # Display the original image, rescale back by 255
    fig, ax = pyplot.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow(A*255)
    ax[0].set_title('Original')
    ax[0].grid(False)
    
    # Display compressed image, rescale back by 255
    ax[1].imshow(X_recovered*255)
    ax[1].set_title('Compressed, with %d colors' % K)
    ax[1].grid(False)    
       
    
    
    
    
    




