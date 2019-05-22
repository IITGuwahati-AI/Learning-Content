from matplotlib import pyplot as plt
from matplotlib import style
import numpy as np
style.use('ggplot')
a=((np.loadtxt('data.txt'))[1:,:])
b=a[:,1:]
y=a[:,0]
i=0
while(i<9):
    j=i+1
    while(j<10):
        x1=b[:,i]
        x2=b[:,j]
        for k in range(999):
            if (y[k]==1):
                plt.scatter(x1[k],x2[k],color='r')
            if (y[k]==2):
                plt.scatter(x1[k],x2[k],color='b')

       
        plt.title('Scattering')
        plt.ylabel('Y axis')
        plt.xlabel('X axis')        
        plt.show()
        j=j+1
    i=i+1      
#the graph between column1 and column 2 can be classified perfectly
           
    

