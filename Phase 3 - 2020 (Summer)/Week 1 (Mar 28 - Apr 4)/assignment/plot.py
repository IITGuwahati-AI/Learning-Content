import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
numpy_data = np.loadtxt("data.txt")
dataset = pd.DataFrame(data = numpy_data,columns =['Labels','1','2','3','4','5','6','7','8','9','10'])
#print(dataset)
label = ['Labels']
X = dataset.loc[:, label].values #separating out the label column
fig= plt.figure(figsize=(10,10))
a =  fig.subplots(3,3)
for i in range(len(dataset)):
    if X[i] == 1.0:
        a[0][0].scatter(dataset['1'][i], dataset['2'][i], color='r')
    if X[i] == 2.0:
        a[0][0].scatter(dataset['1'][i], dataset['2'][i], color='b')
a[0][0].set_xlabel('feature 1', fontsize = 5)
a[0][0].set_ylabel('feature 2', fontsize = 5)
a[0][0].set_title('PLOT', fontsize = 7)
a[0][0].grid()


for i in range(len(dataset)):
    if X[i] == 1.0:
        a[0][1].scatter(dataset['2'][i], dataset['3'][i], color='r')
    if X[i] == 2.0:
        a[0][1].scatter(dataset['2'][i], dataset['3'][i], color='b')
a[0][1].set_xlabel('feature 2', fontsize = 5)
a[0][1].set_ylabel('feature 3', fontsize = 5)
a[0][1].set_title('PLOT', fontsize = 7)
a[0][1].grid()


for i in range(len(dataset)):
    if X[i] == 1.0:
        a[0][2].scatter(dataset['3'][i], dataset['4'][i], color='r')
    if X[i] == 2.0:
        a[0][2].scatter(dataset['3'][i], dataset['4'][i], color='b')
a[0][2].set_xlabel('feature 3', fontsize = 5)
a[0][2].set_ylabel('feature 4', fontsize = 5)
a[0][2].set_title('PLOT', fontsize = 7)
a[0][2].grid()



for i in range(len(dataset)):
    if X[i] == 1.0:
        a[1][0].scatter(dataset['4'][i], dataset['5'][i], color='r')
    if X[i] == 2.0:
        a[1][0].scatter(dataset['4'][i], dataset['5'][i], color='b')
a[1][0].set_xlabel('feature 4', fontsize = 5)
a[1][0].set_ylabel('feature 5', fontsize = 5)
a[1][0].set_title('PLOT', fontsize = 7)
a[1][0].grid()


for i in range(len(dataset)):
    if X[i] == 1.0:
        a[1][1].scatter(dataset['5'][i], dataset['6'][i], color='r')
    if X[i] == 2.0:
        a[1][1].scatter(dataset['5'][i], dataset['6'][i], color='b')
a[1][1].set_xlabel('feature 5', fontsize = 5)
a[1][1].set_ylabel('feature 6', fontsize = 5)
a[1][1].set_title('PLOT', fontsize = 7)
a[1][1].grid()

for i in range(len(dataset)):
    if X[i] == 1.0:
        a[1][2].scatter(dataset['6'][i], dataset['7'][i], color='r')
    if X[i] == 2.0:
        a[1][2].scatter(dataset['6'][i], dataset['7'][i], color='b')
a[1][2].set_xlabel('feature 6', fontsize = 5)
a[1][2].set_ylabel('feature 7', fontsize = 5)
a[1][2].set_title('PLOT', fontsize = 7)
a[1][2].grid()


for i in range(len(dataset)):
    if X[i] == 1.0:
        a[2][0].scatter(dataset['7'][i], dataset['8'][i], color='r')
    if X[i] == 2.0:
        a[2][0].scatter(dataset['7'][i], dataset['8'][i], color='b')
a[2][0].set_xlabel('feature 7', fontsize = 5)
a[2][0].set_ylabel('feature 8', fontsize = 5)
a[2][0].set_title('PLOT', fontsize = 7)
a[2][0].grid()


for i in range(len(dataset)):
    if X[i] == 1.0:
        a[2][1].scatter(dataset['8'][i], dataset['9'][i], color='r')
    if X[i] == 2.0:
        a[2][1].scatter(dataset['8'][i], dataset['9'][i], color='b')
a[2][1].set_xlabel('feature 8', fontsize = 5)
a[2][1].set_ylabel('feature 9', fontsize = 5)
a[2][1].set_title('PLOT', fontsize = 7)
a[2][1].grid()


for i in range(len(dataset)):
    if X[i] == 1.0:
        a[2][2].scatter(dataset['9'][i], dataset['10'][i], color='r')
    if X[i] == 2.0:
        a[2][2].scatter(dataset['9'][i], dataset['10'][i], color='b')
a[2][2].set_xlabel('feature 9', fontsize = 5)
a[2][2].set_ylabel('feature 10', fontsize = 5)
a[2][2].set_title('PLOT', fontsize = 7)
a[2][2].grid()

plt.show()
