import numpy as np
import matplotlib.pyplot as plt

filename = "data.txt"
loadtxt = np.loadtxt(filename, delimiter='\t',skiprows=1)
data = loadtxt
a=[]
b=[]

for i in range(0,999):
    if int(data[i][0])==1:
        a.append(data[i])
    else:
        b.append(data[i])

a=np.asarray(a)   
b=np.asarray(b)
a=np.transpose(a)
b=np.transpose(b)

for i in range(1,11):
    for j in range(i,11):
        
        if i==j:
            continue
        x=a[i]
        y=a[j]
        plt.plot(x,y, color ='red')
        
        u=b[i]
        v=b[j]
        plt.plot(u,v,color ='blue')
        plt.grid(True, color='black')
        
        plt.title('Graphs')
        plt.xlabel('Feature '+str(i))
        plt.ylabel('Feature '+str(j))
        plt.show()