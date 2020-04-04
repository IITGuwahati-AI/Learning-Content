import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt

data = np.loadtxt('data.txt', delimiter = '\t', skiprows = 1 )

label = data [:,0]
features = np.zeros((999,10))
features = data[:,1:]
scaled_data = preprocessing.scale(features.T)
pca = PCA() # create a PCA object
pca.fit(scaled_data) # scales the paramater

pca_data = pca.transform(scaled_data) # get PCA coordinates for scaled_data
pca_percentage = np.round(pca.explained_variance_ratio_* 100, decimals=1)
print(pca_percentage)

principalComponents = pca.fit_transform(features)

for i in range (1,999) :
	if (label[i] == 1) :
		plt.scatter(principalComponents[i,0],principalComponents[i,1],color='r',marker = 'x')
	else:
		plt.scatter(principalComponents[i,0],principalComponents[i,1],color='b',marker = 'o')
plt.title('PCA')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.savefig('PCA')

