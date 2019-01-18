import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
data = np.genfromtxt("data.txt",skip_header=1)
label1=[data[1]]
label2=[data[0]]
for i in range(2,data.shape[0]):
	if data[i,0]==1:
		label1=np.concatenate((label1,[data[i]]))
	else:
		label2=np.concatenate((label2,[data[i]]))



for i in range(1,10):
	for j in range(i+1,11):
		plt.plot(label1[:,i],label1[:,j],'b')
		plt.plot(label2[:,i],label2[:,j],'r')
		plt.xlabel(str(i)+' column')
		plt.ylabel(str(j)+' column')
		plt.title('graph between '+str(i)+' '+str(j)+' columns')
		plt.show()

