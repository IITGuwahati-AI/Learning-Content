import numpy as np
from matplotlib import pyplot as plt
data = np.loadtxt('data.txt',skiprows = 1)#imports txt file and skips 1st row
#print(data)
red=np.array([x for x in data if int(x[0])==1])#array for all label 1

blue=np.array([x for x in data if int(x[0])==2])#array for all label 2

#for loop for plotting all combination of graphs
for i in range(1,11):

    for j in range(i+1,11):
        print(i,j)
        #scatter plot
        plt.scatter(red[:,i],red[:,j],c='r',s=1)#c is for color s is for dot size

        plt.scatter(blue[:,i],blue[:,j],c='b',s=1)

        #labels for axes
        plt.xlabel("feature "+str(i))
        plt.ylabel("feature "+str(j))
        #title of plot
        plt.title("feature plot "+str(j)+" vs "+str(i))
        plt.savefig('./plots/'+"plot"+str(i) + '-' + str(j))

        plt.show()#shows plot

        
print("THE TWO LABELS ARE CLASSIFIED IN A PLOT BETWEEN FEATURES 1 AND 2")
