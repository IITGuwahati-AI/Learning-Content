import numpy as np
from matplotlib import pyplot as plt
data = np.loadtxt('../data.txt',delimiter='\t',skiprows=1)
for p in range(1,10):
    for q in range(1,10):
        if p<q:
          for i,j in enumerate(data[:,[0]]):
              if j==1:
                  plt.scatter(data[i,p],data[i,q],s=5,color='r')
              elif j==2:
                  plt.scatter(data[i,p],data[i,q],s=5,color='b')
          plt.title('F'+str(p)+'vs'+'F'+str(q))
          plt.ylabel('feature'+str(q))
          plt.xlabel('feature'+str(p))
          plt.show()
                  
                 
