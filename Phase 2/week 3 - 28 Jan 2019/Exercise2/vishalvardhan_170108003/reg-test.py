from logistic_learn import *

data = np.loadtxt('2.txt', delimiter = ',')
X = data[:, :2]
y = data[:, 2]

#visualize the data
plotData(X, y)
# Labels and Legend
pyplot.xlabel('Microchip Test 1')
pyplot.ylabel('Microchip Test 2')

# Specified in plot order
pyplot.legend(['y = 1', 'y = 0'], loc='upper right')
pyplot.savefig('reg-data-plot')
pyplot.show()
#=================================================

# Note that mapFeature also adds a column of ones for us, so the intercept
# term is handled
X = utils.mapFeature(X[:, 0], X[:, 1])

# Initialize fitting parameters
initial_theta = np.zeros(X.shape[1])
# Set regularization parameter lambda to 1
# DO NOT use `lambda` as a variable name in python
# because it is a python keyword
lambda_ = 1

# Compute and display initial cost and gradient for regularized logistic
# regression
cost, grad = costFunctionReg(initial_theta, X, y, lambda_)

print('Cost at initial theta (zeros): {:.3f}'.format(cost))
print('Expected cost (approx)       : 0.693\n')

print('Gradient at initial theta (zeros) - first five values only:')
print('\t[{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}]'.format(*grad[:5]))
print('Expected gradients (approx) - first five values only:')
print('\t[0.0085, 0.0188, 0.0001, 0.0503, 0.0115]\n')


# Compute and display cost and gradient
# with all-ones theta and lambda = 10
test_theta = np.ones(X.shape[1])
cost, grad = costFunctionReg(test_theta, X, y, 10)

print('------------------------------\n')
print('Cost at test theta    : {:.2f}'.format(cost))
print('Expected cost (approx): 3.16\n')

print('Gradient at initial theta (zeros) - first five values only:')
print('\t[{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}]'.format(*grad[:5]))
print('Expected gradients (approx) - first five values only:')
print('\t[0.3460, 0.1614, 0.1948, 0.2269, 0.0922]')
#=================================================

print('------------------------------\n')
for lambda_ in range(11):
	options= {'maxiter': 400}
	res = optimize.minimize(costFunctionReg,
	                        initial_theta,
	                        (X, y, lambda_),
	                        jac=True,
	                        method='TNC',
	                        options=options)

	# the fun property of `OptimizeResult` object returns
	# the value of costFunction at optimized theta
	cost = res.fun

	# the optimized theta is in the x property
	theta = res.x

	utils.plotDecisionBoundary(plotData, theta, X, y)
	pyplot.xlabel('microchp-test1')
	pyplot.ylabel('microchp-test2')
	pyplot.legend(['y = 1', 'y = 0'])
	pyplot.grid(False)
	pyplot.title('lambda = %0.2f' % lambda_)
	pyplot.savefig("decision-boundary-" + str(lambda_))
	pyplot.show()
	# Compute accuracy on our training set
	p = predict(theta, X)

	print('Train Accuracy (with lambda = {:0.2f}): {:.1f}'.format(lambda_,(np.mean(p == y) * 100)))

