import numpy as np
#from matplotlib import pyplot as plt                # For plots only
data_array = np.loadtxt('data.txt', delimiter='\t', skiprows=1)

list_lab1 = []                                       # empty list for label 1
list_lab2 = []                                       # empty list for label 2



for i in range(999):                             # Random variable i
    if( data_array[i, 0]==1):
        list_lab1.append(data_array[i, 1:])
    else:
        list_lab2.append(data_array[i, 1:])
        
array_lab1 = np.array(list_lab1)                    # Required array of data of label 1
array_lab2 = np.array(list_lab2)                    # Required array of data of label 2


print(array_lab1)
print(array_lab2)
"""
#For the plots
for x in range(0,10):                              # For some random variable x and y
    for y in range(x, 10):
         if x != y :
             plt.title('Feature {} vs {}'.format(x, y))
             plt.scatter(array_lab1[:, x], array_lab1[:, y], color ='r', label='Label 1')
             plt.scatter(array_lab2[:, x], array_lab2[:, y], color ='b', label='Label 2')
             plt.xlabel('Feature {}'.format(x))
             plt.ylabel('Feature {}'.format(y))
             plt.legend()
             plt.show()
"""
         


"""
Created on Sun Apr  2 01:05:21 2020

@author: mailm
"""