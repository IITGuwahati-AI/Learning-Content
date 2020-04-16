import os
import numpy as np
from matplotlib import pyplot

a = np.loadtxt('data.txt', skiprows = 1)
for i in range(1 , 11):
    for j in range(1 , 11):
        if(i != j):
            Y = a[:,i]
            X = a[:,j]
            for k in range(0, 999):
                if a[k,0] == 1 :
                    pyplot.plot(X[k],Y[k],'ro')
                    pyplot.title('FEATURE: {} VS FEATURE: {}'.format(i,j))
                    pyplot.ylabel('FEATURE: {}'.format(i))
                    pyplot.xlabel('FEATURE: {}'.format(j))
                else :
                    pyplot.plot(X[k],Y[k],'bo')
                    pyplot.title('FEATURE: {} VS FEATURE: {}'.format(i,j))
                    pyplot.ylabel('FEATURE: {}'.format(i))
                    pyplot.xlabel('FEATURE: {}'.format(j))
        pyplot.show();