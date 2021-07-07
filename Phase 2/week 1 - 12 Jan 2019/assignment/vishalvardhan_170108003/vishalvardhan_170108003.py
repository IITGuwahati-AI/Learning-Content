import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('../data.txt',skiprows = 1)
#all rows of label 2 ,labelled as red
#list object later convert to numpy array
red1 = np.array([x for x in data if int(x[0]) == 1])
#red1.shape = (500,11)
#similarily
blue2 = np.array([x for x in data if int(x[0]) == 2])
#blue2.shape = (499,11)
#ist to matrix asarray(list_object) returns np.array()
domain1 = np.linspace(1,10,500,dtype = int)
domain2 = np.linspace(1,10,499,dtype = int)
number_of_plots = 0

for i in range(1,11):
	for j in range(i+1,11):
		#plotting the feature vs feature plot using scatter to check if the 
		number_of_plots += 1
		plt.scatter(red1[:,i],red1[:,j],label = 'red',color = 'r',s = 2)
		plt.scatter(blue2[:,i],blue2[:,j],label = 'blue',color = 'b' ,s = 2)
		plt.title('features-plot')
		x_name = 'feature-' + str(j)
		y_name = 'feature-' + str(i) 
		plt.xlabel(x_name)
		plt.ylabel(y_name)
		#the following line can be used to save plots to a particular location
		#plt.savefig('./plots/'+str(i) + '-' + str(j))
		plt.legend()
		plt.show()

print(f"number of plots checked = {number_of_plots}")
print("best features are 1 and 2")

