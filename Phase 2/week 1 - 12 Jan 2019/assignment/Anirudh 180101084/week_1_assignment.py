import numpy as np
from matplotlib import pyplot as plt
data = np.genfromtxt("data.txt", skip_header=1)
n = 1
while(n<10):
    m = n+1
    while(m<=10):
        plt.figure()
        label1_x = []
        label2_x = []
        label1_y = []
        label2_y = []
        for p in range(999):
            if data[p,0] == 1:
                label1_x.append(data[p,n])
            else:
                label2_x.append(data[p,n])
            if data[p,0] == 1:
                label1_y.append(data[p,m])
            else:
                label2_y.append(data[p,m])
        plt.scatter(label1_x,label1_y, color = 'r',label = 'label 1')
        plt.scatter(label2_x,label2_y,color = 'b', label = 'label 2')
        plt.xlabel('feature '+str(n))
        plt.ylabel('feature '+str(m))
        plt.title('feature '+str(n)+' vs feature '+str(m)+' graph')
        plt.legend()
        plt.savefig('feature '+str(n)+' vs feature '+str(m))
        m = m+1
    n = n+1
