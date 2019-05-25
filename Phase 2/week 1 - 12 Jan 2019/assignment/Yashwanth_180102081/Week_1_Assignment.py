import numpy as np
import matplotlib.pyplot as plt
d = np.loadtxt('data.txt', skiprows=1)
for i in range(1,11):
    for j in range(1,11):
        if i<j:
          for k,lablel in enumerate(d[:,[0]]):
              if lablel==1:
                  plt.scatter(d[k,i],d[k,j],s=1,color='r')
              elif lablel==2:
                  plt.scatter(d[k,i],d[k,j],s=1,color='b')
          plt.title('F'+str(i)+'vs'+'F'+str(j))
          plt.ylabel('Feature'+str(j))
          plt.xlabel('Feature'+str(i))
          plt.show()
