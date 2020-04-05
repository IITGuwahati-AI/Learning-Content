import numpy as np    #for handling the data
import matplotlib.pyplot as plt  # plot purpose
from matplotlib import style   # its type of styles in matplotlib
data = np.genfromtxt('data.txt' , delimiter='\t')
print(data.shape)
print(data)
a1 = np.zeros((999,11))
a2 = np.zeros((999,11))
i = 0
j = 0
for rows in data[1: , :]:
            if 1 == rows[0]:
                  a1[i , :]  = rows
                  i += 1
            if 2 == rows[0]:   
                   a2[j , :]  = rows
                   j += 1
for i in range (1 , 10):
        for j in range (2 , 11):
           if i < j :
             a = b = A = B =  np.zeros((999,1))
             a = a1[:, i] 
             b = a1[:, j]
             A = a2[:, i] 
             B = a2[:, j]
             style.use('ggplot')

             plt.scatter(a,b , c='r', label='Label = 1')
             plt.scatter(A,B , c='b', label='Label = 2')

             plt.title('Graph Between %d vs %d Feature' %(j , i))
             plt.ylabel('Y axis is Feature %d' %j)
             plt.xlabel('X axis is Feature %d' %i)

             plt.legend()


#Showing what we plotted
             plt.show()
            # Pair of Feature 1 and Feature 2 is  the two features which can classify the two labels perfectly.



















