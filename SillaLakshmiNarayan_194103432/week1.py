import numpy as np
from matplotlib import pyplot as plt

#importing data using numpy
datas = np.loadtxt('data.txt',delimiter = '\t', skiprows=1)

label = datas[:,0]
data = datas[:,1::]

rows,columns=data.shape
print(rows,columns)


#plotting feature vs feature
for i in range(0,columns):				# i is for x axis feature
	for j in range(i+1,columns):			# j is for y axis feature 
		plt.figure('feature %d vs feature %d'%(i+1,j+1))
		plt.title('feature %d vs feature %d'%(i+1,j+1))
		plt.xlabel('feature %d'%(i+1))
		plt.ylabel('feature %d'%(j+1))
		

		for k in range(0,rows):						# k is for rows, so k,i make x plot and k,j make y plot

			if label[k]==1:
				plt.scatter(data[k,i],data[k,j],color='r')
			else:
				plt.scatter(data[k,i],data[k,j],color='b')
		plt.savefig('feature %d vs feature %d.png'%(i+1,j+1))			
		#plt.show()

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from statistics import variance


#scaling of data
new_data = StandardScaler().fit_transform(data)


#10D data to a 2D data
pca = PCA(n_components=2)

#axis of PCA i.e. 10 dimension data reduced to 2 dimension
principalComponents = pca.fit_transform(new_data)

print(pca.explained_variance_ratio_,pca.explained_variance_)   #total % of data retained will be 

#components of pca are eigen vectors of covariance matrix and eigen vectors arranged in decresing order
#magnitude represent the weightage to a perticular feature andindex to maximum magnitude represent the feature with most contribution
ev=pca.components_  #ev for eigen vectors
#print(ev)

ev1=ev[0,:]
ev2=ev[1,:]
print(ev1,ev2)

result1=np.amax(np.abs(ev1))
result2=np.amax(np.abs(ev2))
print(result1,result2)
#after running max value of pca 1 is at 3rd place and max value of pca 2 is at 10

print('best features according to pca are 3 and 10')

pca_1=principalComponents[:,0]
pca_2=principalComponents[:,1]

abs_pca_1=np.abs(pca_1)
abs_pca_2=np.abs(pca_2)
print(abs_pca_1,abs_pca_1)
#visualization
plt.figure('pca')
plt.xlabel('pca 1')
plt.ylabel('pca 2')

for i in range(0,rows):
	if label[i]==1:
		plt.scatter(principalComponents[i,0],principalComponents[i,1], color='r')
	else:
		plt.scatter(principalComponents[i,0],principalComponents[i,1], color='b')

plt.savefig('pca features')
plt.show()

#with pca feature 3 and 10 are obtained best and without pca 1 and 2