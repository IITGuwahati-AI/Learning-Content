import numpy as np
import matplotlib.pyplot as plt


dataFile='data.txt'
data=np.loadtxt(dataFile,delimiter='\t',skiprows=1)

f1,f2,f3,f4,f5,f6,f7,f8,f9,f10=[],[],[],[],[],[],[],[],[],[]
F1,F2,F3,F4,F5,F6,F7,F8,F9,F10=[],[],[],[],[],[],[],[],[],[]

features1=[f1,f2,f3,f4,f5,f6,f7,f8,f9,f10]
features2=[F1,F2,F3,F4,F5,F6,F7,F8,F9,F10]

for j in range(1,11):
    for i in range(999):
        x=data[i,j]
        if data[i,0]==1:
            features1[j-1].append(x)
        else:
            features2[j-1].append(x)
k=0
for i in range(10):
    for j in range(i+1,10):
        plt.figure(k)
        plt.scatter(features1[i],features1[j],label='label 1',c='r')
        plt.scatter(features2[i],features2[j],label='label 2',c='b')
        plt.title('Feature %d vs. Feature %d' %(j+1,i+1))
        plt.xlabel('Feature '+str(i+1))
        plt.ylabel('Feature '+str(j+1))
        plt.legend()
        k+=1
        plt.show()
        if input()=='y':        ##input 'y' to see the next plot else, the program ends
            flag=1
            break
        else:
            flag=0
    if flag==1:
        break
    
    
