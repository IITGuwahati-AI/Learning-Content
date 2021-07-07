import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
numpy_data = np.loadtxt("data.txt")
dataset = pd.DataFrame(data = numpy_data,columns =['Labels','1','2','3','4','5','6','7','8','9','10'])
#print(dataset)
values = ['1','2','3','4','5','6','7','8','9','10']
label = ['Labels']
X = dataset.loc[:, label].values #separating out the label column
Y = dataset.loc[:, values].values #separating out the values
Y = StandardScaler().fit_transform(Y) #Standardising the values
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(Y)
principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
finalDf = pd.concat([dataset[['Labels']],principalDf] ,axis = 1)
print(finalDf)
#plt.plot(principalComponents, "*")
#plt.show()

plt.xlabel('Principal Component 1', fontsize = 15)
plt.ylabel('Principal Component 2', fontsize = 15)
plt.title('2 component PCA', fontsize = 20)
for i in range(len(dataset)):
    if X[i] == 1.0:
        plt.scatter(principalDf['principal component 1'][i], principalDf['principal component 2'][i], color='r')
    if X[i] == 2.0:
        plt.scatter(principalDf['principal component 1'][i], principalDf['principal component 2'][i], color='b')
plt.grid()
plt.show()
