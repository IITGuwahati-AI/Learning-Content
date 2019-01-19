# -*- coding: utf-8 -*-

import numpy as np; 
import matplotlib.pyplot as plt;

def plot_graph(x, y,i,j,c): 
    # plotting the actual points as scatter plot 
    plt.scatter(x, y, color = c, marker = "o", s = 10) 
    

    plt.xlabel("feature "+str(i)) 
    plt.ylabel("feature "+str(j)) 
    
    #plt.axhline(0, linestyle="dashed")
    #plt.axvline(0, linestyle="dashed")
    #plt.axis('scaled')
    # function to show plot 
    plt.show() 

if __name__=="__main__":
    arr = np.loadtxt(fname="data.txt",skiprows=1);
    arr = arr.T;
    (graph_no,label_no)=arr.shape;
    
    a=list(arr[0]);
    b=list();
    for i in range(0,label_no):
        if a[i] == np.float64(1):
            a[i]='r';
        else:
            a[i]='b';
    
    for i in range(1,graph_no):
        if i+1 < graph_no:
            for j in range(i+1,graph_no):
                plot_graph(arr[i],arr[j],i,j,a);
