import numpy as np
from matplotlib import pyplot as plt
data = np.loadtxt('../data.txt',delimiter='\t',skiprows=1)
for i in range(1,11):
    for j in range(1,11):
        if i<j:
          for lable1,lable2 in enumerate(data[:,[0]]):
              if lable2==1:
                  plt.scatter(data[lable1,i],data[lable1,j],s=3,color='r')
              elif lable2==2:
                  plt.scatter(data[lable1,i],data[lable1,j],s=3,color='b')
          plt.title('F'+str(i)+'vs'+'F'+str(j))
          plt.ylabel('Feature'+str(j))
          plt.xlabel('Feature'+str(i))
          plt.show()

