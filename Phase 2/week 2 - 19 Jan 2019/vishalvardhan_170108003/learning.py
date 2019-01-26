import numpy as np
import matplotlib.pyplot as plt
import time

def comp_cost(x,y,theta):
	theta = np.array(theta)
	theta = np.append(theta, -1)
	mat = np.vstack((np.ones(len(x)), x.T))
	mat = np.vstack((mat, y))
	cost = np.matmul(theta, mat)
	cost = np.matmul(cost,cost.T)
	cost /= (len(x) * 2)
	return cost

def plot_func(x, y, mode = 'plot',color = 'g', x_label = 'population', y_label = 'profit', title = None, label = None):
	if mode == 'plot':
		plt.plot(x,y,label = label, color = color)
	elif mode == 'scatter':
		plt.scatter(x,y,label = label, color = color, marker = '.')
	else:
		plt.hist(x,y,label = label, color = color)
	plt.title(title)
	plt.xlabel(x_label)
	plt.ylabel(y_label)
	if label:
		plt.legend()

def step_grad(x,y,theta,alpha):
	if type(theta) == list:	
		theta = np.array(theta)
	alpha = alpha/len(x)
	line = np.ones(len(x))
	mul = np.vstack((line,x.T))
	return theta - np.matmul((np.matmul(theta, mul) - y),(mul.T))*alpha

def final_des(x,y,theta,alpha):
	print('wait for 15 seconds before the output shows up')
	theta = np.array(theta)
	minim = None
	final_theta = theta
	start = time.time()
	while(1):
		if time.time() - start > 15:
			break
		theta = step_grad(x,y,theta,alpha)
		val = comp_cost(x,y,theta)
		if (not minim) or val < minim:
			minim = val
			final_theta = theta
			if val == 0:
				break
	# print(f"cost value after descent : {minim}")			
	return final_theta			

def feat_normalise(mat):
	mat = mat.T
	n = len(mat[0])
	mean = []
	for x in mat:
		mean.append(sum(x)/n)
	mean = np.array(mean)
	mean = np.repeat(mean.reshape(len(mean), 1), n, axis = 1)
	mat = mat - mean
	for x in mat:
		try:
			x = x/(np.matmul(x, x)/n)
		except:
			pass
	return mat.T

def normal_equation(x, y):
	mul = np.vstack((np.ones(len(x)), x.T))
	return np.matmul(np.linalg.inv(np.matmul(mul,mul.T)), np.matmul(mul,y.reshape(len(y), 1))).T[0]