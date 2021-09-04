from matplotlib import pyplot as plt

import numpy as np


label = np.loadtxt('C:/Users/Sanskar Kumar/Desktop/data.txt',usecols =(0),skiprows = 1)
for j in range(1,11):
    x = np.loadtxt('C:/Users/Sanskar Kumar/Desktop/data.txt', usecols=(j),skiprows = 1)
    for i in range(j+1,11):
        y= np.loadtxt('C:/Users/Sanskar Kumar/Desktop/data.txt',usecols =(i),skiprows = 1)
        for k in range(0,998):

            if (label[k]== 1):
                plt.scatter(x[k],y[k],s=3, color='red')
            elif (label[k]==2) :
                plt.scatter(x[k],y[k], s=3, color='blue')

        plt.title('Feature' + str(j) + ' vs ' + 'Feature' + str(i))
        plt.ylabel('Feature' + str(i))
        plt.xlabel('Feature' + str(j))

        plt.show()
