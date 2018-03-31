import numpy as np
import matplotlib.pyplot as plt
c='data.txt'
x=np.loadtxt(c,delimiter='	',skiprows=1)
#	DELIMITER HERE IS A TAB SPACE

#	FOR DRAWING ALL POSSIBLE PLOTS
 
#for j in range (1,11):
#	for k in range (1,11) :
#		for i in range (1,999) :
#			if (x[i,0] == 1) :
#				plt.scatter(x[i,j],x[i,k],color='g',marker = 'x')
#			elif (x[i,0] == 2) :
#				plt.scatter(x[i,j],x[i,k],color='y',marker = 'o')
#		print(j)
#		print(k)
#		plt.show()


for i in range (1,999) :
	if (x[i,0] == 1) :
		plt.scatter(x[i,1],x[i,2],color='r',marker = 'x')
	elif (x[i,0] == 2) :
		plt.scatter(x[i,1],x[i,2],color='b',marker = 'o')
plt.title('features')
plt.xlabel('1st attribute')
plt.ylabel('2nd attribute')

plt.show()



