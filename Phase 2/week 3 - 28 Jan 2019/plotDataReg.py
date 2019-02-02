import numpy as np 
import matplotlib.pyplot as plt 

data=np.loadtxt('./ex2data2.txt',delimiter=',')
x=data[:,[0,1]]
y=data[:,2]

plt.plot(x,y,'k+')
plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')
plt.legend(['y = 1', 'y = 0'], loc='upper right')
plt.show()