import numpy as np

from matplotlib import pyplot as plt

# plt.style.use('classic')

array=np.loadtxt("data.txt")

label=np.zeros((998))

feature_x=np.zeros((998))
feature_y=np.zeros((998))

list=[[0,0],[0,1],[0,2],[1,0],[1,1],[1,2],[2,0],[2,1],[2,2]]


for i in range(1,999):
	label[i-1]=array[i][0]

fig,ax = plt.subplots(nrows=3,ncols=3)


for i in range(1,10):

	t=list[i-1][0]
	p=list[i-1][1]

	for j in range(1,999):
		feature_x[j-1]=array[j][i]
		feature_y[j-1]=array[j][i+1]


	for k in range(998):
		if label[k]==1 :
			ax[t][p].scatter(feature_x[k],feature_y[k],color='r')
		else:
			ax[t][p].scatter(feature_x[k],feature_y[k],color='b')


	ax[0][1].set_title('My_Plot')
	ax[t][p].set_xlabel('feature '+str(i))
	ax[t][p].set_ylabel('feature '+str(i+1))
	

plt.show()
	







