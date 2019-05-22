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

var = pca.explained_variance_ratio_
print(var)
# prints the float type array which contains variance ratios for each principal component.
# var = [0.23726612 0.17676103 0.15981101 0.09621538 0.09111189 0.08667718 0.07268585 0.04182227 0.01873639 0.01269897]

