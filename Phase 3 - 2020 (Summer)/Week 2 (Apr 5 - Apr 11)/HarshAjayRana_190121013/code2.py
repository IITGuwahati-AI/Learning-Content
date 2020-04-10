import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
df = pd.read_csv("ex1data1.txt", sep=",",  header=None, names=["Population", "Profit"])
#print(df)
X = df["Population"]
Y = df["Profit"]
Y_pred = []
M = []
B = []
iteration = []
cost = []

#initialising the variables
m = 0
b = 0
alpha = 0.01
epochs = 1500

#our learning algorithm
N = float(len(Y))
for i in range(epochs):
    iteration.append(i)
    Y_pred = (m * X) + b
    m = m - (alpha * ((1/N) * sum(X * (Y_pred-Y))))
    b = b - (alpha * ((1/N) * sum(Y_pred-Y)))
    M.append(m)
    B.append(b)
    costly = (sum((M[i] * X) + B[i]-Y)**2)/(2*epochs)
    cost.append(costly)
M = np.array(M)
B = np.array(B)
cost = np.array(cost)
iteration = np.array(iteration)


#plotting

x1 = np.linspace(5,22.5,97)
y = m*x1 + b
plt.plot(x1, y)
plt.plot(X, Y, "x", color = "r")
plt.xlabel("Population of city in 10,000s")
plt.ylabel("Profit in $10,000s")
plt.show()

plt.plot(iteration, cost)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.show()


#using sklearn to print the score
X1 = np.asarray(X)
X1 = X1.reshape(len(X), 1)
#print(X1)
reg = LinearRegression()
reg = reg.fit(X1, Y)
Y_pred = reg.predict(X1)
R_score = reg.score(X1, Y)
print("The R score is", R_score)
