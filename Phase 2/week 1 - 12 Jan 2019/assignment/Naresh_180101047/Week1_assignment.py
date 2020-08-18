import numpy as np
import matplotlib.pyplot as plt
j=1
data = np.loadtxt('../data.txt', skiprows=1)
for i,label in enumerate(data[:,[0]]) :
    if label==1 :
	    plt.scatter(data[i,[1]], data[i,[2]], s=3, c='red')
    elif label==2 :
	    plt.scatter(data[i,[1]], data[i,[2]], s=3, c='blue')

plt.xlabel('Feature-1')
plt.ylabel('Feature-2')
plt.title('F1 vs F2')
plt.show()