from matplotlib import pyplot as plt
import numpy as np
data=np.genfromtxt("data.txt")
lbl1=data[data[:,0]==1] #makes an array of all rows having 1 as their label
lbl2=data[data[:,0]==2]
figure=plt.figure(1)
cols=3
rows=3
#Number of rows and columns in the figure canvas
n=10 #number of features
i=1  
for x in range(1,n+1):
    for y in range(x+1,n+1):
        #This double loop setup allows x and y to iterate over all possible combinations of feature vs feature graphs
        figure.add_subplot(rows,cols,i)
        fx1=lbl1[:,x]
        fy1=lbl1[:,y]
        fx2=lbl2[:,x]
        fy2=lbl2[:,y]
        plt.scatter(fx1,fy1,color='r')
        plt.scatter(fx2,fy2,color='b')
        plt.xlabel("Feature "+str(x))
        plt.ylabel("Feature "+str(y))   
        if(i%(cols*rows)==0): #For capping the number of subplots in one figure
            plt.tight_layout()
            plt.show()
            print("Press enter")
            figure=plt.figure(i) #Creates a fresh figure to populate with the next subplots
            input() #To provide a break
            i=0
        i=i+1
if(i!=1): #if some plots are left to be shown  
    plt.tight_layout()
    plt.show()