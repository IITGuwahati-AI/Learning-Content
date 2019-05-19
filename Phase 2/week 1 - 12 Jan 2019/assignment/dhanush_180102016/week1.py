import numpy as np
import matplotlib.pyplot as plt

#feature on x-axis
x=1	
#feature on y-axis
y=2

#loading the data
df_list=np.loadtxt("data.txt",skiprows=1,delimiter="\t")


r1x=[]
r2x=[]
r1y=[]
r2y=[]
for i in range(1,999):
	if(df_list[i][0]==1):
		r1x.append(df_list[i][x])
		r1y.append(df_list[i][y])
	elif (df_list[i][0]==2):
		r2x.append(df_list[i][x])
		r2y.append(df_list[i][y])

#plotting
plt.plot(r1x,r1y,'.',c='r',label="Label=1")
plt.plot(r2x,r2y,'.',c='b',label="Label=2")
plt.legend()
plt.title("WEEK 1")
plt.xlabel("Feature "+str(x))
plt.ylabel("Feature "+str(y))
plt.show()

'''
The best plot would be the plot between the features 1 and 2. The plot is included in the folder.
In this plot, the red dots and blue dots are perfectly seperated by a boundary.
'''
