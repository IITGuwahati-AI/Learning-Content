
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

dataframe=pd.read_csv('data.txt', sep='\t')

x= dataframe.iloc[:,1:].values
y=dataframe.iloc[:,0:1].values
x= StandardScaler().fit_transform(x)
pca=PCA(n_components=10)
principalComponents=pca.fit_transform(x)
a=pca.explained_variance_ratio_
print(a)

for i in range(0,10):
	for j in range(i+1,10):
		x1=x[:,i:i+1]
		x2=x[:,j:j+1]
		f=np.concatenate((x1,x2),axis=1)
		pca=PCA(n_components=1)
		principalComponents1=pca.fit_transform(f)
		k=pca.explained_variance_ratio_
		print(str(i+1)+' '+str(j+1)+' '+str(k))
