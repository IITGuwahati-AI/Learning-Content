import numpy as np
from matplotlib import pyplot as plt
data = np.loadtxt('data.txt',delimiter='\t',skiprows=1)
for i,j in enumerate(data[:,[0]]):
   if j==1:
       plt.scatter(data[i,9],data[i,8],s=5,color='r')
   elif j==2:
       plt.scatter(data[i,9],data[i,8],s=5,color='b')
plt.title('F9 vs F8')
plt.ylabel('Feature 8')
plt.xlabel('Feature 9')
plt.show()