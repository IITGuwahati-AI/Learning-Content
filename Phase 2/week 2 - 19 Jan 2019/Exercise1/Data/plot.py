import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
data=np.genfromtxt("ex1data1.txt",delimiter=',')
population=data[:,0]
profit=data[:,1]
model = LinearRegression(fit_intercept=True)

model.fit(population[:, np.newaxis], profit)
xfit = np.linspace(0, 25, 50)
yfit = model.predict(xfit[:, np.newaxis])
plt.xlabel('population')
plt.ylabel('profit')
plt.title('population vs profit')
plt.scatter(population,profit,color='blue')
plt.plot(xfit,yfit,color='red')
plt.show()
yval=model.predict(population[:,np.newaxis])
cost=0
for i in range(len(population)):
	cost+=(yval[i]-profit[i])**2
cost/=2*len(population)
print(f'after applying gradient descent we get min value of cost function as {cost}')
print(f'values of theta0 and theta1 respectively are {model.intercept_} , {model.coef_[0]}')




