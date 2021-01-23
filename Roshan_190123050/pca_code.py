import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import sklearn.preprocessing as pr
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt 

dataset=pd.read_csv('data.txt', delim_whitespace=True)
df=dataset
idx_1 = np.where(dataset['Label'] == 1)
idx_2 = np.where(dataset['Label'] == 2)
df1=df.iloc[idx_1]
df2=df.iloc[idx_2]
c=['1','2','3','4','5','6','7','8','9','10']
x = df.loc[:, c].values
x=StandardScaler().fit_transform(x)
pca=PCA(n_components=2)
principal_components=pca.fit_transform(x)
principalDf=pd.DataFrame(data=principal_components,columns=['Principal components 1','Principal components 2'])

plt.scatter(principalDf.iloc[idx_1]['Principal components 1'], principalDf.iloc[idx_1]['Principal components 2'], s=10, c='r', marker="o", label='Label 1')
plt.scatter(principalDf.iloc[idx_2]['Principal components 1'], principalDf.iloc[idx_2]['Principal components 2'], s=10, c='b', marker="o", label='Label 2')


plt.xlabel('Principal component 1')
plt.ylabel('Principal component 2')
plt.legend()
plt.show()