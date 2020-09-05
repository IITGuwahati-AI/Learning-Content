import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('data.txt', skiprows=1)
for j,label in enumerate(data[:,[0]]) :
    
        if label==1 :
                 plt.scatter(data[j,[2]], data[j,[10]], s=4, c='red')
        elif label==2 :
	         plt.scatter(data[j,[2]], data[j,[10]], s=4, c='blue')
plt.xlabel('Feature-2')
plt.ylabel('Feature-10')
plt.title('F2 vs F10')
plt.savefig('f2vsf10.png')
plt.show()