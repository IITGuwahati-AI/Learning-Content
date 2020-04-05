import numpy as np
from matplotlib import pyplot as plt
from matplotlib import style

style.use('fivethirtyeight')

data=np.loadtxt("data.txt", skiprows=1, unpack=True)

for i in range(1,10):
    for j in range(i+1,11):
        x_data=data[:,i]
        y_data=data[:,j]
        plt.plot(x_data,y_data, linewidth=1,  color='b')
        plt.show()


