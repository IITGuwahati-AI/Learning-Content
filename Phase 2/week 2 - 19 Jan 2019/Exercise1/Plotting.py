import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("/home/abhishek/ex1data1.txt", delimiter=',')

x= data[:,0]
y= data[:,1]


plt.plot(x,y,'rx')
plt.ylabel('Profit in $10,000s')
plt.xlabel('Population in 10,000s')
plt.show()
