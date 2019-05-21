import numpy as np
import matplotlib.pyplot as plt


long_list=[]
counter=0
result = np.loadtxt('data.txt',delimiter='	',skiprows = 1)
one = np.array([x for x in result if int(x[0]) == 1])
two = np.array([x for x in result if int(x[0]) == 2])
for i in range(1,10):
	for j in range(i+1,11):
		plt.xlabel('Feature-'+str(i))
		plt.ylabel('Feature-'+str(j))
		plt.title("Plot between" +str(i)+' and '+str(j) + ' features')
		plt.scatter(one[:,i],one[:,j],c='r',marker='.')
		plt.scatter(two[:,i],two[:,j],c='b',marker='.')		
		plt.savefig('./Jaswanth_170102048/'+ str(i)+'&'+str(j))
		plt.show()
		plt.clf()
