#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 11:45:37 2020

@author: pankaj
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


data_set=np.loadtxt('../data.txt', skiprows=1)

for i in range(1,data_set.shape[1]):
    for j in range(1,data_set.shape[1]):
        if i>j or i==j:
            continue
        

        l=j+1
        m=j-i
        print(i,l,m)
        X1=data_set[:,i:l:m]
        x_scaled=StandardScaler().fit(X1)
        x_data=x_scaled.transform(X1)
        
        
        pca = PCA()
        pca.fit(x_data)
        x_pca = pca.transform(x_data)
        
        plt.clf()
        fig=plt.figure(figsize=(10,5))
        ax1=fig.add_subplot(2, 2, 1)
        ax2=fig.add_subplot(2, 2, 2)
        
        ax1.scatter(x_pca[:,0],x_pca[:,1],c=data_set[:,0],cmap='rainbow')
        ax1.set(xlabel='PC1', ylabel='PC2')
        ax1.set_title('PC plot')
        
        per_var = np.round(pca.explained_variance_ratio_* 100, decimals=1)
        labels = ['PC' + str(x) for x in range(1, len(per_var)+1)]
        ax2.bar(x=range(1,len(per_var)+1), height=per_var, tick_label=labels)
        ax2.set(xlabel='Principal Component', ylabel='Percentage of Explained Variance')
        ax2.set_title('Scree Plot')
        fname='feature ' + str(i) + ' vs '+ str(j)
        plt.savefig('PCA_plots/'+fname)