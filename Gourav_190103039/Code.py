import numpy as  np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

data=np.genfromtxt('Data.txt',delimiter='\t',skip_header=1)
print(np.shape(data))

label=data[:,0]
print(label.dtype)
label=label.astype('int32')
print(label.dtype)
features=data[:,1:]

label1=np.array([]).reshape(0,10)
label2=np.array([]).reshape(0,10)
for i in range(999):
    if label[i]==1:
        label1=np.vstack((label1,features[i,:]))
    else:
        label2=np.vstack((label2,features[i,:]))

for i in range(10):
    for j in range(i+1,10):
        x1=np.array(label1[:,i])
        y1=np.array(label1[:,j])
        x2=np.array(label2[:,i])
        y2=np.array(label2[:,j])
        plt.scatter(x1,y1,label='1')
        plt.scatter(x2,y2,label='2')
        plt.legend()
        plt.ylabel('Feature:{}'.format(j+1))
        plt.xlabel('Feature:{}'.format(i+1))
        plt.title('Feature:{} vs Feature:{}'.format(j+1,i+1))
        plt.show()    
