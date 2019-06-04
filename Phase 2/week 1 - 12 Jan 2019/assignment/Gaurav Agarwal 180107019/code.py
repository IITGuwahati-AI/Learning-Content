import numpy as np 
from matplotlib import pyplot as plt 
data = np.loadtxt('data.txt',skiprows = 1)
l1=np.array(data[np.nonzero(data[:,0]==1.0)])
l2=np.array(data[np.nonzero(data[:,0]==2.0)])

for i in range(1,11):

    for j in range(i+1,11):
        print(i,j)
        
        plt.scatter(l1[:,i],l1[:,j],c='r',s=1)
    
        plt.scatter(l2[:,i],l2[:,j],c='b',s=1)
    
       
        plt.xlabel("feature "+str(i))
        plt.ylabel("feature "+str(j))
       
        plt.title("feature plot "+str(j)+" vs "+str(i))
        plt.savefig('./plots/'+"plot"+str(i) + '-' + str(j))

        plt.show()
