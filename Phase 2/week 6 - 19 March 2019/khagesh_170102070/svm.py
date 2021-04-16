import os
import numpy as np
import re
from matplotlib import pyplot
from scipy import optimize
from scipy.io import loadmat
import utils
grader = utils.Grader()

def gaussianKernel(x1, x2, sigma):
    sim = 0
    # ====================== YOUR CODE HERE ======================

    sim = np.exp((-np.sum(np.power(x1-x2,2)))/(2*sigma**2))

    # =============================================================
    return sim

def dataset3Params(X, y, Xval, yval):
    # You need to return the following variables correctly.
    C = 1
    sigma = 0.3

    # ====================== YOUR CODE HERE ======================
    c={ 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30}
    s={ 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30}
    best_prediction = 100000000.0
    for C in c:
        for sigma in s:
            model = utils.svmTrain(X, y, C, gaussianKernel, args=(sigma,))
            predictions = utils.svmPredict(model, Xval)
            current_prediction = np.mean(predictions != yval)
            if current_prediction < best_prediction:
                best_prediction = current_prediction
                final_C = C
                final_sigma = sigma
    
    # ============================================================
    #return C, sigma
    return final_C, final_sigma

if __name__=='__main__':
    # Load from ex6data1
    # You will have X, y as keys in the dict data
    if 1<0:
        data = loadmat(os.path.join('Data', 'ex6data1.mat'))
        X, y = data['X'], data['y'][:, 0]
        
        # Plot training data
        utils.plotData(X, y)
        
        # You should try to change the C value below and see how the decision
        # boundary varies (e.g., try C = 1000)
        c={1,100}
        for C in c:
            #C = 1
            model = utils.svmTrain(X, y, C, utils.linearKernel, 1e-3, 20)
            utils.visualizeBoundaryLinear(X, y, model)
            pyplot.title('SVM Decision boundary for C='+str(C))
            pyplot.show()
            
        x1 = np.array([1, 2, 1])
        x2 = np.array([0, 4, -1])
        sigma = 2
        
        sim = gaussianKernel(x1, x2, sigma)
        
        print('Gaussian Kernel between x1 = [1, 2, 1], x2 = [0, 4, -1], sigma = %0.2f:'
              '\n\t%f\n(for sigma = 2, this value should be about 0.324652)\n' % (sigma, sim))
        
        
        
        # Load from ex6data2
        # You will have X, y as keys in the dict data
        data = loadmat(os.path.join('Data', 'ex6data2.mat'))
        X, y = data['X'], data['y'][:, 0]
        
        # Plot training data
        utils.plotData(X, y)
        pyplot.show()
        
        # SVM Parameters
        C = 1
        sigma = 0.1
        
        model= utils.svmTrain(X, y, C, gaussianKernel, args=(sigma,))
        utils.visualizeBoundary(X, y, model)
        pyplot.show()
    
    
    # Load from ex6data3
    # You will have X, y, Xval, yval as keys in the dict data
    data = loadmat(os.path.join('Data', 'ex6data3.mat'))
    X, y, Xval, yval = data['X'], data['y'][:, 0], data['Xval'], data['yval'][:, 0]
    
    # Plot training data
    utils.plotData(X, y)
    pyplot.show()
    
    # Try different SVM Parameters here
    C, sigma = dataset3Params(X, y, Xval, yval)
    
    # Train the SVM
    # model = utils.svmTrain(X, y, C, lambda x1, x2: gaussianKernel(x1, x2, sigma))
    model = utils.svmTrain(X, y, C, gaussianKernel, args=(sigma,))
    utils.visualizeBoundary(X, y, model)
    print(C, sigma)
    pyplot.show()
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    