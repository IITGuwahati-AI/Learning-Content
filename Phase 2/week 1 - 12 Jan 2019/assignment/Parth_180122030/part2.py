import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


df = pd.read_csv("../data.txt", sep="\t", header =0)

features =[]
for i in range(1,11,1):
    features.append(str(i))

X=df.loc[:,features]

X = StandardScaler().fit_transform(X)

y = df.loc[:,'Label']
#prit(y)

pca =   PCA (n_components =2)
principalcomponents = pca.fit_transform(X)
principalDf = pd.DataFrame(data=principalcomponents,columns = ['principal component 1','principal component 2'])

finalDf = pd.concat([principalDf,df[['Label']]], axis = 1)

#print(finalDf)

from matplotlib import pyplot as plt
from matplotlib import style
style.use('ggplot')


"""targets = ['1','2']
colors = ['r', 'b']
for target in targets:
    for color in colors:
        indicesToKeep = finalDf['Label'] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1'], finalDf.loc[indicesToKeep, 'principal component 2'], c = color, s = 50)
ax.legend(targets)
ax.grind()"""

ind1 = finalDf['Label']=='1'
ind2 = finalDf['Label']=='2'
plt.scatter(finalDf.loc[ind1,'principal component 1'],finalDf.loc[ind1, 'principal component 2'], color = 'red')
plt.scatter(finalDf.loc[ind2,'principal component 1'],finalDf.loc[ind2, 'principal component 2'], color = 'blue')
plt.title('f0vf1?')
plt.xlabel('principal1')
plt.ylabel('principal2')
plt.show()
