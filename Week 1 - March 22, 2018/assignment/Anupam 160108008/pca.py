#import sklearn.decomposition
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as mpl
from sklearn.preprocessing import StandardScaler

input=np.loadtxt('data.txt',skiprows=1)

print(input[:,1:])

scaledata=input[:,1:]

print(scaledata)

pca=PCA(n_components=2)
principalComponents = pca.fit_transform(scaledata)

#principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
#finalDf = pd.concat([principalDf, input[:,0:1], axis = 1)
#final =np.concatenate((input[:,0],principalComponents),axis=1)
print(pca.explained_variance_ratio_)
row_index=0
for label in input[:,0]:
    mpl.plot(principalComponents[row_index,0],principalComponents[row_index,1],'r.' if label==1 else 'b.',markersize=3)
    row_index=row_index+1
mpl.show()
