import os
import numpy as np
from matplotlib import pyplot
import matplotlib as mpl
from scipy import optimize
from scipy.io import loadmat
import utils
grader = utils.Grader()

def estimateGaussian(X):
    # Useful variables
    m, n = X.shape

    # You should return these values correctly
    mu = np.zeros(n)
    sigma2 = np.zeros(n)

    # ====================== YOUR CODE HERE ======================

    mu = np.mean(X,axis=0)
    sigma2 = np.mean(np.power(X-mu,2),axis=0)
    
    # =============================================================
    return mu, sigma2

def selectThreshold(yval, pval):
    bestEpsilon = 0
    bestF1 = 0
    F1 = 0
   
    for epsilon in np.linspace(1.01*min(pval), max(pval), 1000):
        # ====================== YOUR CODE HERE =======================

        tp = np.sum((yval==1) & (pval<epsilon))
        fp = np.sum((yval==0) & (pval<epsilon))
        fn = np.sum((yval==1) & (pval>epsilon))
        precision = tp/(tp+fp)
        recall = tp/(tp+fn)
        F1 = (2*precision*recall)/(precision+recall)

        # =============================================================
        if F1 > bestF1:
            bestF1 = F1
            bestEpsilon = epsilon

    return bestEpsilon, bestF1

if __name__=='__main__':
    #  The following command loads the dataset.
    data = loadmat(os.path.join('Data', 'ex8data1.mat'))
    X, Xval, yval = data['X'], data['Xval'], data['yval'][:, 0]
    
    #  Visualize the example dataset
    pyplot.plot(X[:, 0], X[:, 1], 'bx', mew=2, mec='k', ms=6)
    pyplot.axis([0, 30, 0, 30])
    pyplot.xlabel('Latency (ms)')
    pyplot.ylabel('Throughput (mb/s)')
    pyplot.show()
    pass

    #  Estimate my and sigma2
    mu, sigma2 = estimateGaussian(X)
    
    #  Returns the density of the multivariate normal at each data point (row) 
    #  of X
    p = utils.multivariateGaussian(X, mu, sigma2)
    
    #  Visualize the fit
    utils.visualizeFit(X,  mu, sigma2)
    pyplot.axis([0, 30, 0, 30])
    pyplot.xlabel('Latency (ms)')
    pyplot.ylabel('Throughput (mb/s)')
    pyplot.tight_layout()
    pyplot.show()
    
    #----------------------------------------------------------------------------------

    pval = utils.multivariateGaussian(Xval, mu, sigma2)

    epsilon, F1 = selectThreshold(yval, pval)
    print('Best epsilon found using cross-validation: %.2e' % epsilon)
    print('Best F1 on Cross Validation Set:  %f' % F1)
    print('   (you should see a value epsilon of about 8.99e-05)')
    print('   (you should see a Best F1 value of  0.875000)')
    print('\n# Outliers found: %d' % np.sum(p < epsilon))
    
    #  Find the outliers in the training set and plot the
    outliers = p < epsilon
    
    #  Visualize the fit
    utils.visualizeFit(X,  mu, sigma2)
    pyplot.axis([0, 30, 0, 30])
    #pyplot.axis([10, 18, 10, 20])
    pyplot.xlabel('Latency (ms)')
    pyplot.ylabel('Throughput (mb/s)')
    pyplot.tight_layout()
    
    #  Draw a red circle around those outliers
    pyplot.plot(X[outliers, 0], X[outliers, 1], 'ro', ms=10, mfc='None', mew=2)
    pass
    pyplot.show()

    #----------------------------------------------------------------------------------

    #  Loads the second dataset. You should now have the
    #  variables X, Xval, yval in your environment
    data = loadmat(os.path.join('Data', 'ex8data2.mat'))
    X, Xval, yval = data['X'], data['Xval'], data['yval'][:, 0]
    
    # Apply the same steps to the larger dataset
    mu, sigma2 = estimateGaussian(X)
    
    #  Training set 
    p = utils.multivariateGaussian(X, mu, sigma2)
    
    #  Cross-validation set
    pval = utils.multivariateGaussian(Xval, mu, sigma2)
    
    #  Find the best threshold
    epsilon, F1 = selectThreshold(yval, pval)
    
    print('Best epsilon found using cross-validation: %.2e' % epsilon)
    print('Best F1 on Cross Validation Set          : %f\n' % F1)
    print('  (you should see a value epsilon of about 1.38e-18)')
    print('   (you should see a Best F1 value of      0.615385)')
    print('\n# Outliers found: %d' % np.sum(p < epsilon))
    













