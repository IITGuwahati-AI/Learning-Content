import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import preprocessing

def Plot_all(label_1,label_2):
    for i in range(1,11):
        for j in range(i+1,11):
            plt.scatter(label_1[:,i],label_1[:,j],color = 'red',alpha = 0.5)
            plt.scatter(label_2[:,i],label_2[:,j],color = 'blue',alpha = 0.5)
            plt.xlabel('Plot ' + str(i))
            plt.ylabel('Plot ' + str(j))
            title = 'Plot of '+str(i)+' vs '+ str(j)
            plt.title (title)
            plt.show()
            plt.clf()

data = np.loadtxt("data.txt", delimiter = "\t", skiprows = 1)
df_data = pd.DataFrame(data)
df_data = df_data[[1,2,3,4,5,6,7,8,9,10]]
#print(df_data.head())
print(df_data.shape)
##print(type(data))
##print(data.shape)
##
label_1 = data[data[:,0]==1]
df_1 = pd.DataFrame(label_1)
df_1 = df_1[[1,2,3,4,5,6,7,8,9,10]]
#print(df_1.head())
#print(df_1.shape)
##print(label_1.shape)
label_2 = data[data[:,0]==2]
df_2 = pd.DataFrame(label_2)
df_2 = df_2[[1,2,3,4,5,6,7,8,9,10]]
#print(df_2.head())
#print(df_2.shape)
##print(label_2.shape)

#Plot_all(label_1,label_2)
scaled_data = preprocessing.scale(df_data.T)
pca = PCA()
pca.fit(scaled_data)
pca_data = pca.transform(scaled_data)

per_var = np.round(pca.explained_variance_ratio_*100,decimals=1)
labels = ['PC' + str(x) for x in range(1,len(per_var)+1)]

plt.bar(x= range(1,len(per_var)+1),height = per_var, tick_label = labels)
plt.ylabel('Percentage of explained value')
plt.xlabel('Principal Component')
plt.title('Scree Plot')
plt.show()
plt.clf()

pca_df = pd.DataFrame(pca_data,index = ['1','2','3','4','5','6','7','8','9','10'], columns = labels)

plt.scatter(pca_df.PC1,pca_df.PC2)
plt.title('PCA Graph')
plt.xlabel('PC1 - {0}%'.format(per_var[0]))
plt.ylabel('PC2 - {0}%'.format(per_var[1]))

for sample in pca_df.index:
    plt.annotate(sample, (pca_df.PC1.loc[sample], pca_df.PC2.loc[sample]))

plt.show()















