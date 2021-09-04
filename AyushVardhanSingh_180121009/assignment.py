import numpy as np
from matplotlib import pyplot as plt

my_data =np.genfromtxt('data.txt',delimiter='\t')

x_list_label1 = [] #column j:label1 goes here
x_list_label2 = [] #column j:label2
y_list_label1 = [] #column 1 with having;label 1
y_list_label2 = [] #column 2 with having; label 2

for k in range(10):
    for j in range(k,10):
        if k==j:
            continue
        for i in range(999):
            if my_data[i,0] == 1:  #collecting all data related to label 1 in x and y list with thierlabels
                x_list_label1.append(my_data[i,j+1])
                y_list_label1.append(my_data[i,k+1])
                
            if my_data[i,0] == 2:#collecting all data related to label 2 in x and y list wit their label
                x_list_label2.append(my_data[i,j+1])
                y_list_label2.append(my_data[i,k+1])
                
        plt.scatter(x_list_label1 , y_list_label1,color ='b',s= 2, label = 'label1') #plot for label 1 with blue scatter
        plt.scatter(x_list_label2 , y_list_label2, color = 'r',s=2, label="label2") #plot for label 2 with red scatter
        
        plt.legend(loc = 'upper center')#putting legend in upper centre
        
        xlabel = 'feature -'+ (str)(j+1)#putting the x and y label
        ylabel = 'feature -'+(str)(k+1)
        
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        
        save_as = xlabel + ' vs ' + ylabel+'.png' #savings the images in png formt
        plt.savefig(save_as)
        
        plt.show()
        
        x_list_label1 = []  #emptying the list
        x_list_label2 = []
        y_list_label1 = [] 
        y_list_label2 = []
