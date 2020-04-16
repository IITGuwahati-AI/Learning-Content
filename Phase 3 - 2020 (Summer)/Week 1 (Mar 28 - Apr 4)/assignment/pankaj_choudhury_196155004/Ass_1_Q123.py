#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 21:01:13 2020

@author: pankaj
"""

import numpy as np
import matplotlib.pyplot as plt

data_set= np.loadtxt('../data.txt',skiprows=1) # loading data and skiping label names

#labels=data_set[:,0] #separating labels
#data=data_set[:,1:] #separating data


label1_data=data_set[np.where(data_set[:,0]==1)]

label2_data=data_set[np.where(data_set[:,0]==2)]

for i in range(1,label1_data.shape[1]):
    for j in range(1,label1_data.shape[1]):
        
        plt.clf()
        
        if i==j:
            continue
        
        plt.scatter(label1_data[:,i],label1_data[:,j],marker='*', label='class 1')
        plt.scatter(label2_data[:,i],label2_data[:,j],marker='o', label='class 2')
        
        plt.xlabel('feature column '+ str(i))
        plt.ylabel('feature column '+ str(j))
        plt.legend()
        fname='feature ' + str(i) + ' vs '+ str(j)
        plt.title(fname)
        plt.savefig('plots/'+fname)
        
#from sklearn.preprocessing import StandardScaler
#
#scaled_data=StandardScaler.fit(data_set[:,1:])