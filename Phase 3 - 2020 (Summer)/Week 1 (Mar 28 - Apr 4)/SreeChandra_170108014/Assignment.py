import numpy as np
from matplotlib import pyplot as plt
from matplotlib import style

filedata = np.loadtxt('data.txt',skiprows = 1,unpack = True,usecols=[])

x = filedata[0,:]
print(len(x))
filedata = filedata[1::1,]

print(filedata)
print(len(filedata))
count = 0
for i in range(10) :
    y = filedata[i,:] 
    k=1
    for j in range(10-i-1) :
        y1 = filedata[i+k,:]
        
        plt.plot([],[],color ='r', label ='feature '+str(i+1))
        plt.plot([],[],color ='b', label ='feature '+str(i+k+1))
        k=k+1
        plt.scatter(x,y,color='r',marker='*')
        
        count=count+1
        #plt.scatter(y,y1,color='c',marker='*')
        plt.scatter(x,y1,color='b',marker='*')
        #plt.plot(y,y1,'-')
       
        plt.title('Feature Comparision')
        plt.ylabel('Feature')
        plt.xlabel('Label')
        plt.legend()
        plt.show()
        
print(count)
    
