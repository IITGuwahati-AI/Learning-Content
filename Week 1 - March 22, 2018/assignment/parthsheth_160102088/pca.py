import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn.decomposition import PCA

data = pd.read_csv('data.txt', delimiter = '\t', skiprows = 1, header = None)

pca = PCA(n_components=2)
#pca.fit(data.loc[:,1:])
pr = pca.fit_transform(data.loc[:,1:]) 

#print(pr[:,0])

color = ["red","red","blue"]
plt.scatter(pr[:,1],pr[:,0], c = data[0], cmap=matplotlib.colors.ListedColormap(color))
plt.show()

#print(data.loc[:,0:2])


