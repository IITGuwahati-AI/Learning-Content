import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import time



filename = "data.txt"
data = np.loadtxt(filename, delimiter="\t", skiprows=1)

b=[]
c=[]

for i in range(0,999):
    if int(data[i][0])==1 :
        b.append(data[i])
    else:
        c.append(data[i])

b=np.asarray(b)
c=np.asarray(c)

b=np.transpose(b)
c=np.transpose(c)



for i in range(1,11):
    for j in range(i,11):
        if i==j:
            continue
        x = b[i]
        y = b[j]
        plt.scatter(x, y,label='label 1', color='red')

        v = c[i]
        w = c[j]
        plt.scatter(v, w,label='label 2', color='blue')
        plt.grid(False, color='black')

        plt.title('YAY')
        plt.xlabel('Feature '+str(i))
        plt.ylabel('Feature '+str(j))
        # plt.savefig('Figures/Feature '+str(i)+' vs '+str(j)+'.png')
        plt.legend()
        plt.show()


for i in range(1,11):
    for j in range(i+1,11):
        x=np.c_[data[:,i],data[:,j]]
        x = StandardScaler().fit_transform(x)

        pca=PCA(n_components=1)

        p = pca.fit_transform(x)

#         print('.m')

        l1=plt.plot(p[data[:,0]==1,0],'r+',label='label 1',color='red')
        l2=plt.plot(p[data[:,0]==2,0],'b+',label='label 2',color='blue')

        plt.xlabel('feature'+str(i))
        plt.ylabel('feature'+str(j))

        plt.legend()
        plt.title('Feature'+str(i)+'vs'+str(j))

        # plt.savefig('PCA Figures/f'+str(i)+'VS'+'f'+str(j)+'.png')

        plt.show()

