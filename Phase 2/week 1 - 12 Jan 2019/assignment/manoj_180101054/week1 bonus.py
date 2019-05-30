# -*- coding: utf-8 -*-
"""
Created on Wed May 29 12:57:16 2019

@author: Manoj
"""


import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
data = pd.read_csv('../data.txt', header = None,skiprows=1,delimiter='\t')
df = np.array(data)
df = scale(df)
my_model = PCA(n_components=10)
my_model.fit(df)
p=my_model.explained_variance_ratio_
for i in range(0,10):
    if p[i]==max(p):
        t=i
p[t]=0
for i in range(0,10):
    if p[i]==max(p):
        k=i
t=t+1
k=k+1
print("best features are feature"+str(t)+"and feature"+str(k))
        
