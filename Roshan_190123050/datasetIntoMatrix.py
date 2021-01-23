import numpy as np 
label,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10=np.loadtxt('data.txt',skiprows=1,unpack=True)
matrixOfData=np.asmatrix([label,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10])
print(matrixOfData)

#code by Roshan_190123050
