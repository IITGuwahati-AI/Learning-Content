import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

#def split(arr, cond):
    #return [arr[cond], arr[~cond]]
matrix=np.loadtxt("data.txt", skiprows=1)
matrix2=np.array(matrix[:,1:])
pca=PCA(2)
pca.fit(matrix2)
#print(pca.components_)
#print(pca.explained_variance_)
B = pca.transform(matrix2)
print(B.shape)
plt.scatter(B[:,0],B[:,1])   
l1=np.array(matrix[np.nonzero(matrix[:,0]==1.0)])
l2=matrix[np.nonzero(matrix[:,0]==2.0)] 