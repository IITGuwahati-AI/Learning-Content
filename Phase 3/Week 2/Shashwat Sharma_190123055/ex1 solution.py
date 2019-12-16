import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

#Single variable linear
data1=pd.read_csv("ex1/ex1data1.txt",names=["Pop","Prof"])
reg=LinearRegression()
xtr1=data1['Pop'].values.reshape(-1,1) # values returns a 1d array here, We want a column vector so reshape
ytr1=data1['Prof'].values.reshape(-1,1)
reg.fit(xtr1,ytr1)
# print(f'h(x)={reg.intercept_}+{reg.coef_}x')
# print(type(reg.intercept_))
# print(type(reg.coef_))
data1.plot(x='Pop',y='Prof', kind='scatter') #Pandas uses pyplot to plot stuff
plt.ylabel('Profits in 10,000s') 
plt.xlabel('Population of City in 10,000s')
xtest=np.linspace(1,25,200) #creates a uniformly spaced array having 200 elements b/w 1 and 25 (inclusive)
ytest=reg.predict(xtest[:,np.newaxis]) #again, need a column vector
plt.plot(xtest,ytest,color="green") #Plots the best fit line obtained by regression
plt.show()

#Multi variable linear
data2=pd.read_csv("ex1/ex1data2.txt",names=["Size","Bedrooms","Price"])
#print(data2.shape)
xtr2=data2[['Size','Bedrooms']].values #This creates a 2 column matrix so need to reshape
ytr2=data2["Price"].values[:,np.newaxis] #[:,np.newaxis] is another way to reshape(-1,1)
regg=LinearRegression()
regg.fit(xtr2,ytr2)
print(reg.intercept_)
print(regg.coef_)
