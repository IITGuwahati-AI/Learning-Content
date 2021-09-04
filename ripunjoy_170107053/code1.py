import numpy as np
import matplotlib.pyplot as ma

#Task1: Import dataset to matrix using numpy
data=np.genfromtxt('/home/ripunjoym/Desktop/ML/data.txt', delimiter='\t',names=True)

#Task2: Plotting graphs
#We keep changing f1 and f2 and ma.ylabel to plot graphs between respective labels
f1=data['9']
f2=data['10']
c=data['Label']

for i in range(0,999):
   if c[i]==1:
      ma.scatter(f1[i],f2[i],color='red')
   else:
      ma.scatter(f1[i],f2[i],color='blue')

ma.xlabel('Feature9')
ma.ylabel('Feature10')

ma.show()
