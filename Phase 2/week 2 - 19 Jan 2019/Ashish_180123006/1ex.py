import numpy as np
import matplotlib.pyplot as plt

alpha = 0.01
iterations = 1500

costOverIterations = np.zeros(iterations, dtype=float)


def computeCost(X, y, theta):
    m = y.size
    h = np.matmul(X, theta)
    diff = h - y
    J = 1.0 / (2 * m) * np.sum(diff.transpose() * diff)
    return J


def gradientDescent(X, y, alpha, iterations, theta=None):

    global costOverIterations

    if theta == None:
        theta = np.zeros((2, 1))

    alpha = float(alpha)
    m = y.size
    theta = np.mat(theta, dtype=float)

    for i in range(iterations):
        diff = X * theta - y
        theta = theta - (alpha / m) * (X.transpose()) * diff

        costOverIterations[i] = computeCost(X, y, theta)

    return theta


def featureNormalize(X):
    """X must not have the extra 1s column"""
    colmeans = X.mean(0)
    ran = X.max(0) - X.min(0)
    for col in range(X.shape[1]):
        X[:, col] -= colmeans.item(col)
        if ran.item(col) != 0:
            X[:, col] /= ran.item(col)

    return X


# Print 5x5 identity matrix
A = np.identity(5, dtype=int)
print(f"A = \n{A}\n\n")

# load data for exercise 1
ex1data1 = np.loadtxt("../Exercise1/Data/ex1data1.txt", dtype=float, delimiter=",")
ex1data2 = np.loadtxt("../Exercise1/Data/ex1data2.txt", dtype=float, delimiter=",")

m = ex1data1.shape[0]

X = np.ones((m, 2), dtype=float)
X[:, 1] = ex1data1[:, 0]
X = np.mat(X)
y = np.mat(ex1data1[:, 1]).transpose()

theta = gradientDescent(X, y, alpha, iterations)
x1, x2 = X[:, 1].min(), X[:, 1].max()
y1, y2 = theta.item(0) + theta.item(1) * x1, theta.item(0) + theta.item(1) * x2

# plotting
plt.subplot(211)
plt.plot(ex1data1[:, 0], ex1data1[:, 1], "rx")
plt.xlabel("Population of city in 10,000s")
plt.ylabel("Profit in $10,000s")
plt.plot(np.array([x1, x2]), np.array([y1, y2]), "b-")

plt.subplot(212)
plt.plot(np.arange(iterations, dtype=int), costOverIterations, "-")
plt.xlabel("number of iterations")
plt.ylabel("cost")

plt.show()

