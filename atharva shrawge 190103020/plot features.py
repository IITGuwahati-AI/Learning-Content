import numpy as np
from io import StringIO
from matplotlib import pyplot as plt    


data = np.loadtxt("data.txt", skiprows=1)
label1= (data[:,0]==1)
label2= (data[:,0]==2)



for i in range(1,11):
    for j in range(i+1,11):
        x1= data[:,i][label1]
        y1= data[:,j][label1]
        x2= data[:,i][label2]
        y2= data[:,j][label2]
                
        plt.scatter(x1, y1,color ='r',label="label 1",linewidth=0.05)
        plt.scatter(x2, y2,color ='b',label="label 2",linewidth=0.05)
        
        plt.title("feature"+str(i)+' vs '+"feature"+str(j))
        plt.ylabel("feature"+str(j))
        plt.xlabel("feature"+str(i))
        plt.legend()
        
        name = str(i)+' vs '+str(j) +'.png'
        plt.savefig(name)
        plt.close()
        

       