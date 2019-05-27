import numpy as np
import matplotlib.pyplot as plt

dataset = np.loadtxt(fname = "/home/abhishek/data.txt", skiprows=1)

for i in range(1,10):
    for j in range(i+1,11):
       for k in range(0,len(dataset)-1):
          if dataset[k,0]== 1:
            plt.scatter(dataset[k,i], dataset[k, j], s=3, color= 'red')
          elif dataset[k,0] == 2:
            plt.scatter(dataset[k,i], dataset[k, j], s=3, color= 'blue')
          
       plt.ylabel('Feature-' + str(j))
       plt.xlabel('Feature-' + str(i))
       plt.title('F' + str(i) + ' vs F' + str(j))
       plt.savefig('feature'+str(i)+'vs'+str(j)+'.png')
       plt.show()


      







