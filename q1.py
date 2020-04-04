import numpy as np
from urllib.request import urlopen
from matplotlib import pyplot as plt, style
url='https://raw.githubusercontent.com/IITGuwahati-AI/Learning-Content/master/Phase%203%20-%202020%20(Summer)/Week%201%20(Mar%2028%20-%20Apr%204)/assignment/data.txt'
raw_data=urlopen(url)
dataset=np.loadtxt(raw_data,skiprows=1)
style.use('ggplot')
len1=np.count_nonzero(dataset[:,0]==1)
len2=999-len1
data1=np.empty((len1,11))
data2=np.empty((len2,11))
l1=-1
l2=-1
for i in range(999):
    if dataset[i,0]==1:
        l1+=1
        data1[l1,:]=dataset[i,:]
    else:
        l2+=1
        data2[l2,:]=dataset[i,:]

# following commented code is to draw plots individually
'''
k=0
for i in range(1,10):
    for j in range(i+1,11):
        k+=1
        plt.figure(k)
        plt.title('feature %d vs feature %d'%(i,j))
        plt.scatter(data1[:,i],data1[:,j],c='r',label='1')
        plt.scatter(data2[:,i],data2[:,j],c='b',label='2')
        plt.legend()
        plt.xlabel('feature %d'%i)
        plt.ylabel('feature %d'%j)
'''        
k=0
for i in range(1,10):
    for j in range(i+1,11):
        k+=1
        plt.subplot(5,9,k)
        plt.scatter(data1[:,i],data1[:,j],c='r')
        plt.scatter(data2[:,i],data2[:,j],c='b')
        plt.xlabel('feature %d'%i)
        plt.ylabel('feature %d'%j)        
       
plt.show()
# On observing graphs one can conclude that features 1 and 2 can classify the two labels perfectly
