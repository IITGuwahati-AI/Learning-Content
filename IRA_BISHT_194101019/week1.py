import numpy as np
from matplotlib import pyplot as plt
from matplotlib import style

style.use('ggplot')

#Importing dataset into a matrix using numpy
f=np.genfromtxt('data.txt',delimiter="\t")
f[0][0]=0
row,column=f.shape

#label wise classification of data.
list1=[]
list2=[]
for i in range(row):
    if(f[i][0]==1):
        list1.append(i)
    if(f[i][0]==2):
        list2.append(i)
        
label1=f[list1,:]
label2=f[list2,:]   


#Plotting all features vs each other 
for i in range(1,column):
    #Creating subplots for each feature
    fig, axs = plt.subplots(3, 3,figsize=(15,15))
    plt.subplots_adjust(hspace=0.3)
    fig.suptitle("Plot for feature-%d" % i)
    x=0
    for j in range(1,column):
        x+=1
	#skip plotting if both features are same
        if(i==j):
            x-=1
            continue
        x-=1;

        f1_l1=label1[1::,i]
        f2_l1=label1[1::,j]
        axs[x//3,x%3].scatter(f1_l1,f2_l1,c="r",label="Label 1")
        
        f1_l2=label2[1::,i]
        f2_l2=label2[1::,j]
        axs[x//3,x%3].scatter(f1_l2,f2_l2,c="b",label="Label 2")

        axs[x//3,x%3].set_xlabel('Feature %d' % i)
        axs[x//3,x%3].set_ylabel('Feature %d' % j)
        axs[x//3,x%3].title.set_text("%d vs %d feature plot" % (i, j))
        axs[x//3,x%3].legend()
        x+=1
    plt.show();
    print('\n\n')
#Features 1 and 2 can classify the two labels perfectly.
print("Features 1 and 2 can classify the two labels perfectly.")

#pca
import pandas as pd
from sklearn.decomposition import PCA
from sklearn import preprocessing


without_header=f[1::,1:]
X=pd.DataFrame(without_header)

header=f[0,1:]
X.columns=header
print(X.shape)

from sklearn.preprocessing import StandardScaler
x = X.loc[:, header].values
x = StandardScaler().fit_transform(x) # normalizing the features
print(x.shape)

pca=PCA(n_components=2)
pca.fit_transform(x)
final=pca.components_
print(final)
print(np.argsort(final))
print('Acoording to array, 8th and 10th features are most suitable')

