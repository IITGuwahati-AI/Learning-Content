import numpy as np
import matplotlib.pyplot as plt
d = np.loadtxt('data.txt', skiprows=1)
for i,label in enumerate(d[:,[0]]) :
    if label==1 :
        plt.scatter(d[i,[1]], d[i,[2]], s=1, color='red')
    elif label==2 :
        plt.scatter(d[i,[1]], d[i,[2]], s=1, color='blue')

plt.xlabel('Feature-1')
plt.ylabel('Feature-2')
plt.title('F1 vs F2')
plt.show()