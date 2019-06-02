import numpy as np
data=np.loadtxt("/home/sanjiit/ml/data.txt",skiprows=1)
#print(a)
###    red=np.array(a[j])
#for i in range(6):
    #print(a[i][0])
#print(a[2][0])
#print(red)
red=np.array([x for x in data if int(x[0])==1])
#print(red)
from matplotlib import pyplot as pyp

blue = np.array([x for x in data if int(x[0]) == 2])

for i in range(1, 11):

    for j in range(i + 1, 11):
        print(i, j)

        pyp.scatter(red[:, i], red[:, j], c='r')

        pyp.scatter(blue[:, i], blue[:, j], c='b')

        pyp.xlabel("feature " + str(i))
        pyp.ylabel("feature " + str(j))

        pyp.title("feature plot " + str(j) + " vs " + str(i))
        pyp.savefig('/home/sanjiit/ml/sanjit_180102066/plots/' + "plot" + str(i) + '-' + str(j))

        pyp.show()



print("feature 1 and 2 classify the labels perfectly")
