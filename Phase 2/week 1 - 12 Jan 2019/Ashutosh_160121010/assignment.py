import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


data = np.genfromtxt('data.txt',skip_header = 1)
print(data)
#print(data.shape)

label=data[:,0]
#count = 0
colors=['red','blue']
for i in range(1,11):
	for j in range(i+1,11):
		plt.scatter((data[:,i]),(data[:,j]),c=label,cmap=matplotlib.colors.ListedColormap(colors))
		plt.xlabel('Feature '+str(i))
		plt.ylabel('Feature '+str(j))
		plt.title('Feature '+str(i)+' vs Feature '+str(j))
		plt.show()

print('Best 2 features which can classify the two labels perfectly : '+str(1)+','+str(2))

a = np.delete(data,0,1)
print(a)
pca = PCA()
a = pca.fit_transform(a)
print(pca.explained_variance_ratio_)
print(a)
b = pca.explained_variance_ratio_
c = np.sort(b)
f1=0
f2=0
for i in range(10):
    if b[i] == c[9]:
        f1 = i+1
        break
for i in range(10):
    if b[i] == c[8]:
        f2 = i+1
        break
print('Best 2 features found with pca : '+str(f1)+','+str(f2))