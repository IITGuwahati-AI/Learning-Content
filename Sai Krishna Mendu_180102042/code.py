# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 14:36:41 2019

@author: Sai Krishna Mendu
"""

import numpy as np
import matplotlib.pyplot as plt

filename="data.txt"


data=np.loadtxt( filename,skiprows=1,delimiter="\t")

for i in range(1,11):
    for  r in range(1,10): 
        for m in range(0,999):
            if i+r<11:
                x=data[m,i]
                y=data[m,i+r]
                z=str(i)
                q=str(i+r)
                plt.xlabel("feature "+z)
                plt.ylabel("feature "+q)
                plt.title("feature "+z+"vs "+q)
                if data[m,0]==2:
                    plt.scatter(x,y,color="blue")  
                else:
                    plt.scatter(x,y,color="red")
            else:
                 break
        plt.savefig("ft."+z+"vs"+q+".png")
        plt.show()
        
"""
Out of all the files i have found that by the graph of feature 1 versus 
feature 2 were perfectly differentiated.I have completed the first three steps
by performing the for loop in python and first created a numpy array.
"""     

   

      
                
            