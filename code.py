
import numpy as np
from matplotlib import pyplot as plt
from io import StringIO
count=0
datafile=open("data.txt")
d=[]
for lines in datafile:
    if count==0:
        count=1     # skipping the first heading
        continue
    c=lines.split('\t')     #splitting on the basis of \t
    d.append(list(map(float, c)))

matrix = np.asarray(d)      #imported the dataset to the matrix
label_1=[]
label_2=[]
for val in matrix:
    if int(val[0])==1: label_1.append(val)
    else : label_2.append(val)
label_1=np.asarray(label_1)
label_2=np.asarray(label_2)
feat=[]
for i in range(1, 11):          #pairing the features to plot
    for j in range(i+1, 11): feat.append((i,j))
    #plotting the features
for (x,y) in  feat:
    x1 = label_1[: , x]
    x2 = label_2[: , x]
    y1 = label_1[: , y]
    y2 = label_2[: , y]
    plt.scatter(x1, y1, color='r', label="label 1")
    plt.scatter(x2, y2, color='b', label="label 2")
    plt.title('comparision')
    plt.ylabel('feature {}'.format(y))
    plt.xlabel('feature {}'.format(x))
    plt.legend()
    #plt.savefig("feature {} vs {}.png".format(y,x))
    #plt.clf()
    plt.show()
print("feature 1 vs 2 can best classify the labels ")
