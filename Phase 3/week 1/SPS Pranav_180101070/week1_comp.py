import numpy as np
import matplotlib.pyplot as plt
import os
data = np.loadtxt(os.path.join('assignment', 'data.txt'), delimiter='\t', skiprows=1)
m=np.size(data[0])
for i in range(1,m):
    for j in range(1,m):
        if i<j:
          for k,l in enumerate(data[:,[0]]):
              if l==1:
                  plt.scatter(data[k,i],data[k,j],s=2,color='r')
              else:
                  plt.scatter(data[k,i],data[k,j],s=2,color='b')
          plt.title('Feature {} vs Feature {}'.format(i,j))
          plt.xlabel('Feature {}'.format(i))
          plt.ylabel('Feature {}'.format(j))
          plt.legend(['Label = 1', 'Label = 2'])
          plt.show()
