import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

def sigmoid(z):
    """
    Computes the sigmoid of z.
    """
    return 1.0 / (1.0 + np.exp(-z));

def predict(Theta1, Theta2, X):
    # Make sure the input has two dimensions
    if X.ndim == 1:
        X = X[None]  # promote to 2-dimensions
    
    # useful variables
    m = X.shape[0]
    num_labels = Theta2.shape[0]

    # You need to return the following variables correctly 
    p = np.zeros(X.shape[0])

    # ====================== YOUR CODE HERE ======================

    X = np.hstack((np.array([np.ones(m)]).T,X));
    hidden_layer = sigmoid(np.hstack((np.array([np.ones(m)]).T,np.dot(X,Theta1.T))))
    output = sigmoid(np.dot(hidden_layer,Theta2.T))
    p = np.argmax(output, axis=1);

    # =============================================================
    return p

def displayData(X, example_width=None, figsize=(10, 10)):
    """
    Displays 2D data stored in X in a nice grid.
    """
    # Compute rows, cols
    if X.ndim == 2:
        m, n = X.shape
    elif X.ndim == 1:
        n = X.size
        m = 1
        X = X[None]  # Promote to a 2 dimensional array
    else:
        raise IndexError('Input X should be 1 or 2 dimensional.')

    example_width = example_width or int(np.round(np.sqrt(n)))
    example_height = n / example_width

    # Compute number of items to display
    display_rows = int(np.floor(np.sqrt(m)))
    display_cols = int(np.ceil(m / display_rows))

    fig, ax_array = plt.subplots(display_rows, display_cols, figsize=figsize)
    fig.subplots_adjust(wspace=0.025, hspace=0.025)

    ax_array = [ax_array] if m == 1 else ax_array.ravel()

    for i, ax in enumerate(ax_array):
        ax.imshow(X[i].reshape(example_width, example_width, order='F'),
                  cmap='Greys', extent=[0, 1, 0, 1])
        ax.axis('off')

def image(X,figsize=(5,5)):
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
    plt.figure(figsize=figsize)
    plt.axis('off');
    arr = np.asarray(graph.T)
    plt.imshow(arr, cmap='gray')
    plt.show()

if __name__=='__main__':    
    #  training data stored in arrays X, y
    data = loadmat(os.path.join('', 'ex3data1.mat'))
    X, y = data['X'], data['y'].ravel()

    # set the zero digit to 0, rather than its mapped 10 in this dataset
    # This is an artifact due to the fact that this dataset was used in 
    # MATLAB where there is no index 0
    y[y == 10] = 0

    # get number of examples in dataset
    m = y.size

    # randomly permute examples, to be used for visualizing one 
    # picture at a time
    indices = np.random.permutation(m)

    # Randomly select 100 data points to display
    rand_indices = np.random.choice(m, 100, replace=False)
    sel = X[rand_indices, :]

    #utils.displayData(sel)
    image(sel);
    
    # Setup the parameters you will use for this exercise
    input_layer_size  = 400  # 20x20 Input Images of Digits
    hidden_layer_size = 25   # 25 hidden units
    num_labels = 10          # 10 labels, from 0 to 9

    # Load the .mat file, which returns a dictionary 
    weights = loadmat(os.path.join('', 'ex3weights.mat'))

    # get the model weights from the dictionary
    # Theta1 has size 25 x 401
    # Theta2 has size 10 x 26
    Theta1, Theta2 = weights['Theta1'], weights['Theta2']

    # swap first and last columns of Theta2, due to legacy from MATLAB indexing, 
    # since the weight file ex3weights.mat was saved based on MATLAB indexing
    Theta2 = np.roll(Theta2, 1, axis=0)
    
    pred = predict(Theta1, Theta2, X)
    print('Training Set Accuracy: {:.1f}%'.format(np.mean(pred == y) * 100))
    
    if indices.size > 0:
        i, indices = indices[0], indices[1:]
        displayData(X[i, :], figsize=(2, 2))
        pred = predict(Theta1, Theta2, X[i, :])
        print('Neural Network Prediction: {}'.format(*pred))
    else:
        print('No more images to display!')

















