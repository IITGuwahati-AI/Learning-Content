import numpy as np
import matplotlib.pyplot as plt 

print("Plotting data with + indicating (y = 1) examples and o ' ...'indicating (y = 0) examples.\n");
data=np.loadtxt('./ex2data1.txt',delimiter=",");
x=data[:,[0,1]]
y=data[:,2]
r1=x[y==1]
plt.plot(r1[:,0],r1[:,1],'k+')
r2=x[y==0]
plt.scatter(r2[:,0],r2[:,1],marker='.')
plt.xlabel('exam 1 score')
plt.ylabel('exam 2 score')
plt.legend(['admitted','Not admitted'])
plt.show()