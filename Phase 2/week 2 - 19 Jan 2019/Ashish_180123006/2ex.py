import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize

calls = 0


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def costFunc(theta, X, y):
    global calls
    calls += 1
    """
    theta -> row vector
    X --> standard form
    y --> row vector
    """
    theta = np.matrix(np.array(theta).ravel()).T
    y = np.matrix(np.array(y).ravel())
    X = np.mat(X)
    m = y.size
    h = sigmoid(X * theta)
    term1 = np.log(h)
    term2 = np.log(1 - h)
    J = -np.sum(y * term1 + (1 - y) * term2) / m

    return J


def gradCostFunc(theta, X, y):
    theta = np.mat(np.array(theta).ravel()).T
    y = np.mat(np.array(y).ravel()).T
    m = y.size

    P = sigmoid(X * theta)

    grad = (X.T * (P - y)) / m

    return grad


def plotDecisionBoundary(X, y, theta):
    def line_y(xl):
        return [-(theta[0] + theta[1] * x) / theta[2] for x in xl]

    x1, y1 = X[np.where(y)[0], 1:].T.tolist()
    x2, y2 = X[np.where(1 - y)[0], 1:].T.tolist()

    xmin, xmax = X[:, 1].min(), X[:, 1].max()
    plt.plot(x1, y1, "go")
    plt.plot(x2, y2, "r+")
    plt.plot([xmin, xmax], line_y([xmin, xmax]), "k-")
    plt.show()


def predict(x, theta):
    return sigmoid(np.sum(np.multiply(np.array(x).ravel(), np.array(theta).ravel())))


data = np.loadtxt("../Exercise2/Data/ex2data1.txt", delimiter=",")

"""takes in raw data"""
X = np.hstack((np.ones((data.shape[0], 1)), data[:, :-1]))
y = data[:, -1]
theta = np.matrix([[0] for _ in range(3)])


theta = optimize.fmin_tnc(
    costFunc,
    np.array([0, 0, 0]),
    fprime=gradCostFunc,
    approx_grad=True,
    epsilon=0.001,
    args=(X, y),
)

print(
    "J at [0,0,0] = {}\nmin J at {} = {}".format(
        costFunc([0, 0, 0], X, y), theta[0], costFunc(theta[0], X, y)
    )
)

theta = theta[0]
# final plot A + Bx + Cy = 0 => y = (-a - bx)/c
plotDecisionBoundary(X, y, theta)

