import numpy as np
import matplotlib.pyplot as plt

data=genfromtxt('data.txt',delimiter='\t',names=True)

f1=data['9']
f2=data['10']
c=data['Label']

for i in range(999):
    if c[i]==1:
        plt.scatter(f1[i],f2[i],c='r')
    else:
        plt.scatter(f1[i],f2[i],c='b')

plt.xlabel('feature1')
plt.ylabel('feature2')


plt.show()
