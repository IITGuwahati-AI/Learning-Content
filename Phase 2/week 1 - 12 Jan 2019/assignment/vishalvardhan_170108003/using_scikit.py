#working
from sklearn import svm
from sklearn.model_selection import train_test_split
import numpy as np

data = np.loadtxt('../data.txt',skiprows = 1)
y = data[:,0]
best = [0,0,0]
for i in range(1,11):
	for j in range(i+1,11):
		x = data[:,[i,j]]
		x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.4, random_state = 0)
		clf = svm.SVC(kernel = 'rbf', gamma = 0.7 , C = 1).fit(x_train, y_train)
		temp = clf.score(x_test, y_test)
		if temp > best[2]:
			best[2] = temp
			best[0],best[1] = i,j
		#here best[2] can be used later for printing.	
print(f"Best features are: feature {best[0]} and feature {best[1]}.")			


