# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 17:31:21 2020

@author: Asus
"""
import matplotlib.pyplot as plt
import os
import numpy as np
data = np.genfromtxt('dt.txt',delimiter='\t')


X,y = data[:,1:],data[:,0]
def plotfeatures(a,b,X,y):
    X_copy = X[1:,:]
    y_copy = y[1:]
    
    
    one = y_copy == 1
    two = y_copy == 2
   
    a-=1
    b-=1
    
    
    plt.plot(X_copy[one,a],X_copy[one,b],'bo')
    plt.plot(X_copy[two,a],X_copy[two,b],'ro')
   
    plt.xlabel('label '+str(a+1))
    plt.ylabel('label '+str(b+1))
    plt.title('Classification using labels '+str(a+1)+' and '+str(b+1))
    plotfeatures(1,2,X,y)
    