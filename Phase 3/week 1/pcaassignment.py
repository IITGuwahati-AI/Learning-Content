import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

data_doc = np.loadtxt('C:/Users/Administrator/Desktop/data.txt',delimiter='\t',skiprows=1)
for i in range(1,11):
    for j in range(i+1,11):
        x=np.c_[data_doc[:,i],data_doc[:,j]]
        x = StandardScaler().fit_transform(x)
        pca=PCA(n_components=1)
        p = pca.fit_transform(x)
        l1=plt.plot(p[data_doc[:,0]==1,0],'m.',label='label1')
        l2=plt.plot(p[data_doc[:,0]==2,0],'y.',label='label2')
       
        plt.ylabel('feature'+str(j))
        plt.xlabel('feature'+str(i))
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,ncol=2, mode="expand", borderaxespad=0.)
        plt.title('PCA:feature'+str(i)+'vs'+str(j),fontweight="bold")
        plt.savefig('PCA:f'+str(i)+'VS'+'f'+str(j)+'.png')
        plt.show() 
