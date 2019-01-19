#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 10:18:14 2019

@author: abhilash
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

data_file = np.loadtxt('/home/abhilash/Desktop/data.txt',delimiter='\t',skiprows=1)

for i in range(1,11):
    for j in range(i+1,11):
        plt.plot(data_file[data_file[:,0]==1,i],data_file[data_file[:,0]==1,j],'r.',label='label1')
        plt.plot(data_file[data_file[:,0]==2,i],data_file[data_file[:,0]==2,j],'b.',label='label2')
        plt.xlabel('feature'+str(i))
        plt.ylabel('feature'+str(j))
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,ncol=2, mode="expand", borderaxespad=0.)
        plt.title('feature'+str(i)+'vs'+str(j),fontweight="bold")
        plt.savefig('f'+str(i)+'VS'+'f'+str(j)+'.png')
        plt.show()
        
for i in range(1,11):
    for j in range(i+1,11):
        x=np.c_[data_file[:,i],data_file[:,j]]
        x = StandardScaler().fit_transform(x)
        pca=PCA(n_components=1)
        p = pca.fit_transform(x)
        l1=plt.plot(p[data_file[:,0]==1,0],'m.',label='label1')
        l2=plt.plot(p[data_file[:,0]==2,0],'y.',label='label2')
        plt.xlabel('feature'+str(i))
        plt.ylabel('feature'+str(j))
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,ncol=2, mode="expand", borderaxespad=0.)
        plt.title('PCA:feature'+str(i)+'vs'+str(j),fontweight="bold")
        plt.savefig('PCA:f'+str(i)+'VS'+'f'+str(j)+'.png')
        plt.show()