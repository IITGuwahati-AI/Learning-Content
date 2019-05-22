import numpy as np
from matplotlib import pyplot as plt

data = np.loadtxt("data.txt", delimiter  = '\t', skiprows = 1)

for col1 in range(1,11):
	for col2 in range(1,11):
		if col1 == col2: continue
		for row,label in enumerate(data[:,0]):
			if label == 1:
				plt.scatter(data[row,col1],data[row,col2], s = 3,color = 'red')
			if label == 2:
				plt.scatter(data[row,col1],data[row,col2], s = 4,color = 'blue')
		plt.title('Feature'+ str(col1) +' vs '+'Feature'+ str(col2))
		plt.xlabel('Feature'+ str(col1))
		plt.ylabel('Feature'+ str(col2))
		plt.show()
				
		
				
				

	
