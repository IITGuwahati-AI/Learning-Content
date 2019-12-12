import csv

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#reading the dataset
df = pd.read_table('data.txt', delim_whitespace=True, names=('label', 'feature_1','feature_2','feature_3',
                                                             'feature_4','feature_5','feature_6','feature_7',
                                                             'feature_8','feature_9','feature_10'))
#dropping the heading row
df=df.drop([0], axis=0)
#df.set_index(list(np.arange(len(df[['label']]))))
#assigning the axis
features=['feature_1','feature_2','feature_3','feature_4','feature_5',
          'feature_6','feature_7','feature_8','feature_9','feature_10']
x = df.loc[:, features].values
y = df.loc[:,['label']].values
#standardizing the data
x = StandardScaler().fit_transform(x)

#using PCA

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'],
                           index=list(np.arange(1,len(df[['label']])+1)))
finalDf = pd.concat([principalDf, df[['label']]], axis = 1)
print(finalDf)

#plotting
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 Component PCA', fontsize = 20)


labels=['1', '2']
colors = ['r', 'b']
for label, color in zip(labels,colors):
    indicesToKeep = finalDf['label'] == label
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)

#plotting the heat map to see how the features mixed up to form the components
ax.legend(labels)
ax.grid()
plt.show()
plt.matshow(pca.components_,cmap='viridis')
plt.yticks([0,1],['1st Comp','2nd Comp'],fontsize=10)
plt.colorbar()
plt.xticks(range(len(features)),features,rotation=65,ha='left')
plt.tight_layout()
plt.show()

#entering the results in a csv file

f=open("best features using PCA.csv", 'w')
writer = csv.writer(f)
writer.writerow(features)
for row in pca.components_:
    writer.writerow(row)
f.close()

print("therefore using PCA it is clear that feature 1 and feature2 contain most of the information "
      "and can be used to classify the two labels")

