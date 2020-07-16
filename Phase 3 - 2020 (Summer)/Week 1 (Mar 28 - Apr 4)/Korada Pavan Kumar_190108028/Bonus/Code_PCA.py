

from matplotlib import pyplot as plt                
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# Code is written to find out the pca graph of ten dimensional data given 

df = pd.read_csv('data.txt',delimiter='\t', names=['Label', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10'])

features = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10']
x =df.loc[:, features].values
y = df.loc[:, ['Label']].values
x = StandardScaler().fit_transform(x)
#print(pd.DataFrame(data = x, columns = features).head())

pca = PCA(n_components =2)
principalComponents = pca.fit_transform(x)
principalDf =pd.DataFrame(data = principalComponents
                          , columns =['principal component 1','principal component 2'])
                          
print(principalDf.head(5))
print(df[['Label']].head())

finalDf = pd.concat([principalDf , df[['Label']]], axis = 1)
print(finalDf.head(1000))

fig =plt.figure(figsize =(8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 Component PCA', fontsize = 20)

targets =[ 1, 2]
colors = ['r', 'g', 'b']
for target, color in zip(targets, colors):
    indicesToKeep = finalDf['Label'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s= 50)
    
ax.legend(targets)
ax.grid()


print(pca.explained_variance_ratio_)
 

# General variance ratio >= 85%; here it is very much less(approx40) than that which 
 # indicates that so much of information is lost; so what I infer is that 
 # principal components chosen should be more than '2'




"""
Created on Sun Apr  5 08:04:06 2020

@author: mailm
"""

