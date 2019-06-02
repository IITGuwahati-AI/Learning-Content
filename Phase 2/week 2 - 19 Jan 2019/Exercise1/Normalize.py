import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("/home/abhishek/ex1data2.txt", delimiter=',')
m=len(data)
p=np.size(data,1)

max= np.amax(data, axis=0)
min= np.amin(data, axis=0)

sum = np.sum(data,axis=0)

for i in range(0,p):
    for j in range(0, m):
        data[j,i] -= sum[i]
        data[j,i] /= (max[i] - min[i])
