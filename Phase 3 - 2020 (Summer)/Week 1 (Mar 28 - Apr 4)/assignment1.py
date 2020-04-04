
import numpy as np
from matplotlib import pyplot as plt

matrix = np.genfromtxt('data.txt', delimiter='\t')

y=np.array(matrix[1:,0:1]==1)

matrix1=np.array([matrix[i+1,1:] for i in range(0,999) if y[i,0]==True])
matrix2=np.array([matrix[i+1,1:] for i in range(0,999) if y[i,0]==False])

for i in range(0,10):
	for j in range(i+1,10):
		f01=matrix1[0:,i:i+1]
		f02=matrix1[0:,j:j+1]
		f11=matrix2[0:,i:i+1]
		f12=matrix2[0:,j:j+1]	
		
		plt.scatter(f01,f02,color='r',label = 'Label 1')
		plt.scatter(f11,f12,color='b',label = 'label 2')
		
		plt.xlabel('Feature '+str(i+1))
		plt.ylabel('Feature '+ str(j+1))
		plt.title('Feature ' + str(j+1) +' vs Feature '+ str(i+1))
		plt.legend()
		plt.show()


