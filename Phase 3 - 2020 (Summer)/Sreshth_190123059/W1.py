import numpy as np 
import matplotlib.pyplot as plt

data= np.loadtxt('text.txt',delimiter='\t',skiprows=1)
# print(data)

for i in range(1,11):
    for j in range(i+1,11):

        plt.scatter(data[data[:,0]==1,i],data[data[:,0]==1,j],color='r',label='L1')
        plt.scatter(data[data[:,0]==2,i],data[data[:,0]==2,j],color='b',label='L2')
        sx='feature {}'.format(i)
        sy='feature {}'.format(j)
        st='feature {} vs feature {}'.format(i,j)
        plt.xlabel(sx)
        plt.ylabel(sy)
        plt.title(st)
        plt.legend()
        plt.show()
       
