from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
%matplotlib inline

df = pd.read_csv('C:/Users/avsin/Desktop/jupyter notebook/p 3.7/ml course assign/week 1/PCA_bonus/data-copy1.csv', sep ='\t')

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df)

scaled_data = scaler.transform(df)

from sklearn.decomposition import PCA

pca =PCA(n_components = 2)

pca.fit(scaled_data)

x_pca = pca.transform(scaled_data)

plt.scatter(x_pca[:,0],x_pca[:,1] ,c = df['Label'])

plt.show()