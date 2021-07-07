import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
url = 'https://raw.githubusercontent.com/mathav95raj/Learning-Content/master/Phase%203%20-%202020%20(Summer)/Week%201%20(Mar%2028%20-%20Apr%204)/assignment/data.txt'
data = np.loadtxt(url, skiprows = 1)
df = pd.DataFrame(data)
df=df.rename(columns={0: "Label"})
df["Label"].replace({1: "A", 2: "B"}, inplace=True)
g = sns.PairGrid(df, hue="Label", vars = df.iloc[:,1:], palette = ['r', 'b'], height = 1, aspect = 1)
g = g.map(plt.scatter, s=1)
g = g.add_legend()
x = df.iloc[:,1:]
y = df.iloc[:,0]
x = StandardScaler().fit_transform(x)
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
finalDf = pd.concat([principalDf, y], axis = 1)
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = ['A', 'B']
colors = ['r', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['Label'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()
for i in range(1,11):
    print(np.dot(principalDf.iloc[:,1],df.iloc[:,i]))
    print(np.dot(principalDf.iloc[:,0],df.iloc[:,i]))