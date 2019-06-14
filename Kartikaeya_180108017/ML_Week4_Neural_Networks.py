import os
import numpy as np
from scipy.io import loadmat
import utils

data = loadmat(os.path.join('Data', 'ex3data1.mat'))
X, y = data['X'], data['y'].ravel()
y[y == 10] = 0
m = y.size
indices = np.random.permutation(m)
rand_indices = np.random.choice(m, 100, replace=False)
sel = X[rand_indices, :]
utils.displayData(sel)


input_layer_size  = 400
hidden_layer_size = 25
num_labels = 10
weights = loadmat(os.path.join('Data', 'ex3weights.mat'))
Theta1, Theta2 = weights['Theta1'], weights['Theta2']
Theta2 = np.roll(Theta2, 1, axis=0)
print(np.shape(Theta1))
print(np.shape(Theta2))

def predict(Theta1, Theta2, X):
    if X.ndim == 1:
        X = X[None]
    m = X.shape[0]
    p = np.zeros(X.shape[0])
    print(m)
    X=np.concatenate([np.ones((m,1)),X],axis=1)
    layer2=np.dot(Theta1,np.transpose(X))
    layer2=np.transpose(layer2)
    layer2=np.concatenate([np.ones((m,1)),layer2],axis=1)
    layer3=np.dot(Theta2,np.transpose(layer2))
    p=np.argmax(layer3,axis=0)
    return p

pred = predict(Theta1, Theta2, X)
print('Training Set Accuracy: %',(np.mean(pred == y) * 100))

if indices.size > 0:
    i, indices = indices[0], indices[1:]
    utils.displayData(X[i, :], figsize=(4, 4))
    pred = predict(Theta1, Theta2, X[i, :])
    print('Neural Network Prediction: {}',pred)
else:
    print('No more images to display!')