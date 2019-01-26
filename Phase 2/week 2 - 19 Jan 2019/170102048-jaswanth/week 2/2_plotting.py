import numpy as np
import matplotlib.pyplot as plt

data=np.loadtxt(open('./ex1data1.txt','rb'),delimiter=',')
#print(data)
X=data[:,0]
Y=data[:,1]
m=np.size(Y)
#print(data.shape,m)
print('Plotting data')

plt.plot(X,Y,'rx','MarkerSize',10)
plt.ylabel('Profit in $10,000s')
plt.xlabel('Population of City in 10,000s');
#plt.savefig('assignment_2-plot')
plt.show()
