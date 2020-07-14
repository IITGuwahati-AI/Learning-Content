import numpy as np
from io import StringIO
from matplotlib import pyplot as plt   
from sklearn.preprocessing import StandardScaler

d = StringIO("M F Z\n1 35 58\n2 32 48\n2 62 41\n1 23 16")   
x = np.loadtxt("data.txt", skiprows=1, usecols=[1,2,3,4,5,6,7,8,9,10])
x_std=StandardScaler().fit_transform(x)
x_covar=np.cov(x_std.T)

eival, eivec=np.linalg.eig(x_covar)
eig_pairs=[(np.abs(eival[i]),eivec[:,i]) for i in range(len(eival))]
tot=sum(eival)

var_exp=[(i/tot)*100 for i in eival]
print("variance percentage are: ", var_exp)
max1=0
max2=-1
max1i=0
count=1
for var in var_exp:
    if max1<var:
        max2=max1
        max1=var
        max2i=max1i
        max1i=count
    elif max1>var and max2<var:
        max2=var
        max2i=count
    count+=1

print("Best two features which can classify the two labels perfectly are: " ,max1i," and ", max2i )
















