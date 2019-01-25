# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 11:16:59 2019

@author: Shreyas S K
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

filename = 'ex1data1.txt'
data = np.array(pd.read_csv(filename))
data = np.vstack((data,[6.1101,17.592]))
X = np.array(data[:,0], dtype = float)
Y = np.array(data[:,1], dtype = float)
theta = np.array(np.zeros([2,1]), dtype = int)
m = len(Y)
X = X.reshape(m,1)
Y = Y.reshape(m,1)

plt.scatter(X,Y, color = 'r')
plt.ylabel('Profit in $10,000s', fontsize = 12)
plt.xlabel('Population of City in 10,000s', fontsize = 12)

reg = LinearRegression()
reg.fit(X,Y)

Y_predict = reg.predict(X)

plt.scatter(X,Y, color = 'r')
plt.plot(X,Y_predict)
plt.ylabel('Profit in $10,000s', fontsize = 12)
plt.xlabel('Population of City in 10,000s', fontsize = 12)

m = reg.coef_
c = reg.intercept_

theta = np.array((c,m))

predict1 = np.dot([1,3.5],theta)
print('For population = 35,000, we predict a profit of %f\n', predict1*10000);
predict2 = np.dot([1,7],theta)
print('For population = 70,000, we predict a profit of %f\n', predict2*10000);
        
        
        
        
        
        
        
        
        
        
        
        
