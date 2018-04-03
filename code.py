import numpy as np
from matplotlib import pyplot as plt

#Importing dataset from 'data.txt' delimited by tab
matrix=np.genfromtxt('data.txt',delimiter='	')



for k in range(1,10):	
	for j in range(k+1,11):
		x = []
		for i in range(1,1000):
			if matrix[i,0]==1:
				x=np.append(x,[matrix[i,k]])	#Slicing k-th column from matrix of label 1
		y=[]
		for i in range(1,1000):
			if matrix[i,0]==1:
				y=np.append(y,[matrix[i,j]])	#Slicing j-th column from matrix of label 1
		a = []
		for i in range(1,1000):
			if matrix[i,0]==2:
				a=np.append(x,[matrix[i,k]]) 	#Slicing k-th column from matrix of label 2
		b=[]
		for i in range(1,1000):
			if matrix[i,0]==2:
				b=np.append(y,[matrix[i,j]])	#Slicing j-th column from matrix of label 2

		
		plt.scatter(a,b)
		plt.scatter(x,y,color='red')
		f='Feature '+(str)(k)
		plt.xlabel(f)
		q='Feature '+(str)(j)
		plt.ylabel(q)
		plt.title(f+' vs '+q)
		plt.legend(['Label 1','Label 2'])
		plt.show()


