import numpy as np
from matplotlib import pyplot as plt

a= np.array(np.loadtxt('data.txt', skiprows=1))
print("enter the 1st feature(column)")
k=int(input())
print("enter the 2nd feature(column)")
j=int(input())
x=a[ : , j]
y=a[ : ,k]
label=a[ : ,0]
for i in range(0,999):
    if(label[i]==1):
        plt.scatter(x[i], y[i], s=1,color='red')
    elif(label[i]==2):
        plt.scatter(x[i], y[i], s=1, color='blue')
plt.title("F"+str(k)+" vs. F"+str(j))
plt.ylabel("Feature "+str(k))
plt.xlabel("Feature "+str(j))
plt.show()