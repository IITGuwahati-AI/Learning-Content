import numpy as np
import matplotlib.pyplot as mpl

def gplot(column_index_one,column_index_two,dataset):
    row_index=0
    for label in dataset[:,0]:
        mpl.plot(dataset[row_index,column_index_one],dataset[row_index,column_index_two],'r.' if label==1 else 'b.',markersize=3)
        row_index=row_index+1

    mpl.xlabel('Feature-'+str(column_index_one))
    mpl.ylabel('Feature-'+str(column_index_two))
    mpl.title('Feature-'+str(column_index_one)+' vs Feature-'+str(column_index_two)+'\n Red-Label 1\nBlue-Label 2')

    filename='Feature-'+str(column_index_one)+' vs Feature-'+str(column_index_two)+'.png'
    mpl.savefig(filename)
    mpl.clf()


input=np.loadtxt('data.txt',skiprows=1)

i=1
while i<input.shape[1]:
    j=i+1
    while j<input.shape[1]:
        gplot(i,j,input)
        j=j+1
    i=i+1
