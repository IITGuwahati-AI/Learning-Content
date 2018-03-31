import numpy as np
import matplotlib.pyplot as plt
x=np.loadtxt('data.txt',delimiter='\t',skiprows=1)

#	DELIMITER HERE IS A TAB SPACE

for i in range (1,999) :
	if (x[i,0] == 1) :
		plt.scatter(x[i,1],x[i,2],color='r',marker = 'x')
	else:
		plt.scatter(x[i,1],x[i,2],color='b',marker = 'o')
plt.title('features')
plt.xlabel('1st attribute')
plt.ylabel('2nd attribute')
plt.show()

l=0

#	FOR DRAWING ALL POSSIBLE PLOTS
for j in range (1,11):
	for k in range (1,11) :
		if (j < k) :
			for i in range (1,999) :
				if (x[i,0] == 1) :
					plt.scatter(x[i,j],x[i,k],color='r',marker = 'x')
				else:
					plt.scatter(x[i,j],x[i,k],color='b',marker = 'o')
			plt.title('features')
			plt.xlabel('1st attribute')
			plt.ylabel('2nd attribute')
			l=l+1
			p=str(l)
			plt.savefig(p)
			plt.clf()
