import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.preprocessing import scale

#Load data set
data = np.loadtxt("data.txt", delimiter  = '\t', skiprows = 1)

#convert it to numpy arrays
X = np.array(data)

#Scaling the values
X = scale(X)

pca = PCA(n_components=10)

pca.fit(X)

print(pca.explained_variance_ratio_)

