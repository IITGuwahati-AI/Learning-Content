import numpy as np
from matplotlib import pyplot as plt

matrix = np.loadtxt('data.txt', delimiter='\t', skiprows=1)

cnt = 1

for j in range(1, 10, 1):
    for k in range(j+1, 11, 1):
        x1 = np.array([])
        y1 = np.array([])
        x2 = np.array([])
        y2 = np.array([])

        for i in range(0, 999, 1):
            if matrix[i][0] == 1.:
                x1 = np.append(x1, matrix[i][j])
                y1 = np.append(y1, matrix[i][k])
            else:
                x2 = np.append(x2, matrix[i][j])
                y2 = np.append(y2, matrix[i][k])

        plt.scatter(x1, y1, color="red")
        plt.scatter(x2, y2, color="blue")

        plt.title('Feature-' + str(j) + ' vs ' + 'Feature-' + str(k))
        plt.xlabel('Feature-' + str(j))
        plt.ylabel('Feature-' + str(k))
        plt.legend(['Label 1', 'Label 2'])
        plt.savefig('outputs/plot '+str(j)+' '+str(k)+'.png')

        plt.clf()
        print('Done '+str(cnt))
        cnt = cnt + 1
