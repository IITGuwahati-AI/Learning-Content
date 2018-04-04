import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
a=np.genfromtxt('data.txt',skip_header=1)
print (a)
label=a[:,0]
colors=['red','blue']
for i in range(1,11):
    for j in range(i+1,11):
        plt.scatter((a[:,i]),(a[:,j]),c=label,cmap=matplotlib.colors.ListedColormap(colors))
        plt.xlabel('Feature '+str(i))
        plt.ylabel('Feature '+str(j))
        plt.title('Feature '+str(i)+' vs Feature '+str(j))
        plt.show()
print('Best 2 features which can classify the two labels perfectly : '+str(1)+','+str(2))
b=np.delete(a,0,1)
print(b)
pca = PCA()
b=pca.fit_transform(b)
print(pca.explained_variance_ratio_)
print(b)
c=pca.explained_variance_ratio_
d=np.sort(c)
f1=0
f2=0
for i in range(10):
    if c[i] == d[9]:
        f1 = i+1
        break
for i in range(10):
    if c[i] == d[8]:
        f2 = i+1
        break
print('Best 2 features found with pca : '+str(f1)+','+str(f2))
