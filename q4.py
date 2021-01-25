import numpy as np
from urllib.request import urlopen
from matplotlib import pyplot as plt, style
from sklearn.decomposition import PCA
url='https://raw.githubusercontent.com/IITGuwahati-AI/Learning-Content/master/Phase%203%20-%202020%20(Summer)/Week%201%20(Mar%2028%20-%20Apr%204)/assignment/data.txt'
raw_data=urlopen(url)
dataset=np.loadtxt(raw_data,skiprows=1)
style.use('ggplot')
len1=np.count_nonzero(dataset[:,0]==1)
len2=999-len1
data1=np.empty((len1,11))
data2=np.empty((len2,11))
l1=-1
l2=-1
for i in range(999):
    if dataset[i,0]==1:
        l1+=1
        data1[l1,:]=dataset[i,:]
    else:
        l2+=1
        data2[l2,:]=dataset[i,:]

pca=PCA(n_components=1)
pca.componrnt= True

k=0
for i in range(1,10):
    for j in range(i+1,10):
        k+=1
        x=np.append(data1[:,i],data2[:,i])
        y=np.append(data1[:,j],data2[:,j])
        z=pca.fit_transform(np.stack((x,y),axis=1))
        plt.scatter(z[:len1],np.repeat(k,len1),c='r',label='1')
        plt.scatter(z[len1:],np.repeat(k,len2),c='b',label='2')
plt.xlabel('Transformed component(RED=1,BLUE=2)')
plt.ylabel('combinations of features in order')
plt.title('1D PCA of combinations of features')
plt.show()

# according to PCA results 1 and 2 features are best at classifying labels.


        



