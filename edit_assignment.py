import pandas as pd 

from matplotlib import pyplot as plt

df=pd.read_csv('data.txt',delimiter='\t')

plt.rcParams['figure.figsize']=(20,10)

fig,ax=plt.subplots(nrows=5,ncols=9)

r=c=0

list=[[0,0],[0,1],[0,2],[0,3],[0,4],[0,5],[0,6],[0,7],[0,8],[1,0],[1,1],[1,2],[1,3],[1,4],[1,5],[1,6],[1,7],[1,8],[2,0],[2,1],[2,2],[2,3],[2,4],[2,5],[2,6],[2,7],[2,8],[3,0],[3,1],[3,2],[3,3],[3,4],[3,5],[3,6],[3,7],[3,8],[4,0],[4,1],[4,2],[4,3],[4,4],[4,5],[4,6],[4,7],[4,8]]

flag=-1

for i in range(1,10):
	for j in range(i+1,11):
		flag=flag+1
		r=list[flag][0]
		c=list[flag][1]
		for k in range(999):
			if df['Label'][k]==1:
				ax[r][c].scatter(df[str(i)][k],df[str(j)][k],color='r')
			elif df['Label'][k]==2:
				ax[r][c].scatter(df[str(i)][k],df[str(j)][k],color='b')

		ax[r][c].set_xlabel('feature '+str(i))
		ax[r][c].set_ylabel('feature '+str(j))
        ax[0][4].set_title('My Plot')

# print(plt.rcParams['figure.figsize'])
plt.show()