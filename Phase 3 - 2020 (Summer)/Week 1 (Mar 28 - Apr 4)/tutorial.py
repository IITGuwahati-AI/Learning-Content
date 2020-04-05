from matplotlib import pyplot as plt
from matplotlib import style
import pandas as pd
import numpy as np


style.use('ggplot')

dataset=pd.read_csv('data.txt',delim_whitespace=True)
idx_1=np.where(dataset['Label']==1)
idx_0=np.where(dataset['Label']==2)
columns=["1","2","3","4","5","6","7","8","9","10"]

for column in columns:
	for x in columns[int(column):]:
		plt.scatter(dataset.iloc[idx_1][column],dataset.iloc[idx_1][x],s=10,c='r',marker="o",label='Label 1')
		plt.scatter(dataset.iloc[idx_0][column],dataset.iloc[idx_0][x],s=10,c='b',marker="o",label='Label 2')
		plt.ylabel('feature '+str(x))
		plt.xlabel('feature '+str(column))
		plt.legend()
		plt.show()
