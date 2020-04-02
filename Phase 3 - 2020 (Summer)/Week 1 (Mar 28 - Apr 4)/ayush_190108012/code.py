import matplotlib
from matplotlib import style
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

d=np.loadtxt('data.txt',delimiter ="\t",skiprows=1)

x1=np.array([])
y1=np.array([])

x2=np.array([])
y2=np.array([])

t=0


for i in range(1,10):
	if (i!=9):	
		fig,axs=plt.subplots(10-i,sharex='all')
	for j in range(i+1,11):
		for k in range(0,999):
			x=d[k,0]
			if (x==2.00):			
				x2=np.append(x2, d[k,i])
				y2=np.append(y2, d[k,j])
			else:
				x1=np.append(x1, d[k,i])
				y1=np.append(y1, d[k,j])

		
		t=t+1
		if (i!=9):		
			axs[j-i-1].plot(x1,y1,color='red')
			axs[j-i-1].plot(x2,y2,color='blue')
		else:
			plt.plot(x1,y1,color='red')
			plt.plot(x2,y2,color='blue')

		t1= "Feature " +str(i)
		t2= str(j)
		t3= t1 + " vs " + t2
		
		if (i!=9):
			
			axs[j-i-1].set(ylabel=t2)
			axs[j-i-1].legend(['Label 1','Label 2'])
		else:
			plt.title(t3)
			plt.xlabel(t1)
			plt.ylabel(t2)
			plt.legend(['Label 1','Label 2'])
	if (i!=9):
		axs[10-i-1].set(xlabel=t1)		
		t1="Feature " +str(i) +" vs other features"
		fig.suptitle(t1)	
	plt.show()

