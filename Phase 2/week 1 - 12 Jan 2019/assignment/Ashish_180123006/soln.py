import numpy as np
import matplotlib.pyplot as plt

mat = np.loadtxt('../data.txt', skiprows=1)
f1 = mat[np.where(mat[:,0] < 1.5)][:, 1:]
f2 = mat[np.where(mat[:,0] >= 1.5)][:, 1:]

NUM_FEATURES = f1.shape[-1]

for i in range(NUM_FEATURES):
    for j in range(i+1, NUM_FEATURES):
        plt.plot(f1[:,i], f1[:, j], 'ro', f2[:,i], f2[:, j], 'bo')
        filename = './images/F{}vsF{}.png'.format(i+1,j+1)
        plt.xlabel('Feature {}'.format(i+1))
        plt.ylabel('Feature {}'.format(j+1))
        plt.title('Feature {} vs {}'.format(i+1,j+1))
        plt.savefig(filename)
        plt.close()
        print('Saved file ', filename)
