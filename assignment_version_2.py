import numpy as np

from matplotlib import pyplot as plt

plt.style.use('classic')

array=np.loadtxt("data.txt")

label=np.zeros((998))

feature_x=np.zeros((998))
feature_y=np.zeros((998))

for i in range(1,999):
	label[i-1]=array[i][0]

for i in range(1,9):

	fig,ax = plt.subplots(nrows=1,ncols=10-i)
	index=0

	for j in range(i,10):

		for k in range(1,999):
			feature_x[k-1]=array[k][i]
			feature_y[k-1]=array[k][j+1]


		for t in range(998):
			if label[t]==1 :
				ax[index].scatter(feature_x[t],feature_y[t],color='r')
			elif label[t]==2 :
				ax[index].scatter(feature_x[t],feature_y[t],color='b')

		ax[(10-i)//2].set_title('My_Plot')
		ax[index].set_xlabel('feature '+str(i))
		ax[index].set_ylabel('feature '+str(j+1))
		index=index+1

	plt.show()

for k in range(998):
	feature_x[k]=array[k+1][9]
	feature_y[k]=array[k+1][10]


for t in range(998):
	if label[t]==1 :
		plt.scatter(feature_x[t],feature_y[t],color='r')
	elif label[t]==2 :
		plt.scatter(feature_x[t],feature_y[t],color='b')

plt.title('My_Plot')
plt.xlabel('feature 9')
plt.ylabel('feature 10')
plt.show()

