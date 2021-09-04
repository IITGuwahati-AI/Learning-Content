import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler



df = pd.read_csv('data1.csv', skiprows = 2 ,names=['Labels','Feature 1','Feature 2','Feature 3','Feature 4','Feature 5','Feature 6','Feature 7','Feature 8','Feature 9','Feature 10'])

features = ['Feature 1','Feature 2','Feature 3','Feature 4','Feature 5','Feature 6','Feature 7','Feature 8','Feature 9','Feature 10']
x = df.loc[:, features].values
y = df.loc[:, ['Labels']].values
x = StandardScaler().fit_transform(x)

pd.DataFrame(data = x, columns = features).head()

pca = PCA()
pca.fit_transform(x)
explained_variance=pca.explained_variance_ratio_
print("The explained variance matrix shows the variance ratios of each component among all the features given. It is printed below")
for k in range(0,10):
    print("Explained variance of FEATURE-{} is {}".format(k+1,explained_variance[k]))

pca2 = PCA(n_components=2)
pca.fit_transform(x)
explained_variance_best=pca.explained_variance_ratio_    
print("Seting the number of components to two we get the variance ratios of the two features which classify the labels best or hold max percentage of the data")
for k in range(0,2):
    print("Explained variance of BEST FEATURE-{} is {}".format(k+1,explained_variance_best[k]))

print("Since the variance ratios of the best features matches with that of FEATURE 1 and FEATURE 2 we conclude that Feature 1 and Feature 2 classify the two levels best and this what is also observed graphically")
