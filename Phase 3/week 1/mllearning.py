import numpy as np
import matplotlib.pyplot as plt
data_doc = np.loadtxt('C:/Users/Administrator/Desktop/data.txt',delimiter='\t',skiprows=1)
for i in range(1,11):
    for j in range(i+1,11):
        plt.plot(data_doc[data_doc[:,0]==1,i],data_doc[data_doc[:,0]==1,j],'r.',label='label1')
        plt.plot(data_doc[data_doc[:,0]==2,i],data_doc[data_doc[:,0]==2,j],'b.',label='label2')
        
        plt.ylabel('feature'+str(j))
        plt.xlabel('feature'+str(i))
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,ncol=2, mode="expand", borderaxespad=0.)
        plt.title('feature'+str(i)+'vs'+str(j),fontweight="bold")
        plt.savefig('f'+str(i)+'VS'+'f'+str(j)+'.png')
        plt.show()






