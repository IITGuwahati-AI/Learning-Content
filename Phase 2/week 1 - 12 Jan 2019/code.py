import numpy as np
from matplotlib import pyplot as plt
data = np.loadtxt('data.txt',delimiter='\t',skiprows = 1)
for a in range(1,11,1):
	for b in range(1,11,1):
		if a<b:
			for i,j in enumerate(data[:,[0]]):
				if j==1:
					plt.scatter(data[i,a],data[i,b],s=5,color='r')
				elif j==2:
				    plt.scatter(data[i,a],data[i,b],s=5,color='b')
			plt.title('F'+str(a)+'vs'+'F'+str(b))
			plt.ylabel('F'+str(a))
			plt.xlabel('F'+str(b))
			plt.show()
				    
