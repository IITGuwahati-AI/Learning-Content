import numpy as np
import matplotlib.pyplot as plt

def featureNormalize(x):
	mean=np.mean(x,axis=0)
	x=x-mean
	sigma=np.std(x,axis=0)
	x[:,0]/=sigma[0]
	x[:,1]/=sigma[1]
	return x,mean,sigma

data=np.loadtxt(open('./ex1data2.txt','rb'),delimiter=',')
x=data[:,0:2]
y=data[:,2]
m=np.size(y)
print('normalizing features...')
[x,mean,sigma]=featureNormalize(x)
x=np.insert(x,0,1,axis=1)

print('normalized x is obtained')