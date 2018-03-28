import numpy as np
import matplotlib.pyplot as plt
import matplotlib

y,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10 = np.loadtxt("data.txt", skiprows = 1, unpack = True)

color = ["red","red","blue"]

plt.scatter(x1,x2, c = y, cmap=matplotlib.colors.ListedColormap(color))

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Scatter plot of data taking 2 features at a time \n Red-Label1 Blue-Label2')
plt.show()

