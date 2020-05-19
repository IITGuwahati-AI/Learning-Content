# coding: utf-8
get_ipython().magic(u'cd c:\\users\\dell\\desktop\\py')
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
for j in range(10):
    for k in range(j):
        plt.show()
        
        for i , value_A in enumerate(A[1:,0]):
            if value_A==1:
                plt.plot(A[i,j+1],A[i,k+1],'ro')
            if value_A==2:    
                plt.plot(A[i,j+1],A[i,k+1],'bo')
            plt.xlabel('feature %d' %(j+1))
            plt.ylabel('feature %d' %(k+1))
            plt.title('feature')
            plt.legend(['label 2', 'label 1'])
            
A = np.genfromtxt('data.txt')
for j in range(10):
    for k in range(j):
        plt.show()
        
        for i , value_A in enumerate(A[1:,0]):
            if value_A==1:
                plt.plot(A[i,j+1],A[i,k+1],'ro')
            if value_A==2:    
                plt.plot(A[i,j+1],A[i,k+1],'bo')
            plt.xlabel('feature %d' %(j+1))
            plt.ylabel('feature %d' %(k+1))
            plt.title('feature')
            plt.legend(['label 2', 'label 1'])
            
