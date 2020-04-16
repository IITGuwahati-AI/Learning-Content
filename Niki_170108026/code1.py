import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

def make_plot(i, j):
	fi = dataset[1:, i]
	fj = dataset[1:, j]

	index = 0

	redx = []
	redy = []

	greenx = []
	greeny = []

	for lab in label:
		if lab == 1:
			redx.append(fi[index])
			redy.append(fj[index])
		else :
			greenx.append(fi[index])
			greeny.append(fj[index])
		index += 1

	style.use('ggplot')
	plt.scatter(redx, redy, color='r', label = 'label 1')
	plt.scatter(greenx, greeny, color='b', label = 'label 2')
	plt.title('f' + str(j) + ' vs f' + str(i))
	plt.xlabel('feature ' + str(i))
	plt.ylabel('feature ' + str(j))
	plt.legend()
	plt.show()

dataset = np.loadtxt('data.txt', delimiter='\t')
label = dataset[1:, 0]

for i in range(1,11):
	j = i + 1
	while j < 11:
		make_plot(i,j)
		j += 1