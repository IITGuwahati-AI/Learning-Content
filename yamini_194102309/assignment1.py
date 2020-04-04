# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 13:41:48 2020

@author: Yamini
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

datas = np.loadtxt('https://raw.githubusercontent.com/IITGuwahati-AI/Learning-Content/master/Phase%203%20-%202020%20(Summer)/Week%201%20(Mar%2028%20-%20Apr%204)/assignment/data.txt',skiprows =1)

temp = 1

label = datas[:,0]
data = datas[:,1:]

# for plotting all the features 2 at a time
for j in range (0,10):
    for k in range(j+1,10):
        plt.figure(temp)
        plt.xlabel(f"Feature {j+1}")
        plt.ylabel(f"Feature {k+1}")
        temp+=1
        for i in range(0,999):
            plt.title(f"Feature {j+1} vs Feature {k+1}")
            if label[i] == 1:
                plt.scatter(data[i,j],data[i,k], c = 'r')
            else:
                plt.scatter(data[i,j],data[i,k], c = 'b')
        plt.savefig(f"Feature {j+1} vs {k+1}.png")
pca = PCA(n_components = 2)
new_data1 = StandardScaler().fit_transform(data)
new_data = pca.fit_transform(new_data1)   #transformed into reduced dimensions
print(pca.explained_variance_ratio_,pca.explained_variance_)  #variances and ratios
l = pca.components_

#checking in which original feature direction the new features have the highest component
print("Important features according to PCA",1+np.argsort(np.abs(l[0,:]))[-1],"and",1+np.argsort(np.abs(l[1,:]))[-1])   

#visualization of new features
plt.figure()
plt.title("PCA features")
plt.xlabel(f"PCA Feature 1")
plt.ylabel(f"PCA Feature 2")
for i in range(0,999):    
    if label[i] == 1:
        plt.scatter(new_data[i,0],new_data[i,1], c = 'r')
    else:
        plt.scatter(new_data[i,0],new_data[i,1], c = 'b')
plt.savefig("PCA.png")