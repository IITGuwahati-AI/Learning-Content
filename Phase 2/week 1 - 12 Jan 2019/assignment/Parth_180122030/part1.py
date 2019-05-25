import numpy as np
from matplotlib import pyplot as plt
from matplotlib import style
data = np.loadtxt('../data.txt',skiprows = 1)

X = data[:,1:].copy()
y = data[:,0].copy()
ind2 = np.where(y==2)
ind1 = np.where(y==1)
style.use('ggplot')
for i in range(X.shape[1]):

    for j in range(i+1,X.shape[1],1):
        x_axis=X[:,i]
        y_axis = X[:,j]
        plt.scatter(x_axis[ind1], y_axis[ind1], color = 'red')
        plt.scatter(x_axis[ind2], y_axis[ind2], color = 'blue')
        plt.title('f'+str(i)+'vf'+str(j))
        plt.ylabel('Feature '+str(j))
        plt.xlabel('Feature '+str(i))
        plt.savefig('f'+str(i)+'vf'+str(j)+'.png')
        plt.show()


#the two features which classify the labels perfectly are feature 1 and 2 (the plots f0vf1.png)
