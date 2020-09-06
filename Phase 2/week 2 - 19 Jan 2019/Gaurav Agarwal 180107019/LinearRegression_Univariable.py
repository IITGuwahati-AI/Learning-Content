#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np
from matplotlib import pyplot as plt


# In[ ]:





# In[16]:


data=np.loadtxt("./ex1data1.txt", delimiter=",")
X=data[:,0]
Y=data[:,1]
m=X.size


# In[17]:


print(data)


# In[60]:


print(data[:,0])


# In[32]:


def plot(x,y):
    ax = plt.figure(figsize=(10,10))
    plt.plot(x, y, 'ro', ms=10, mec='k')
    plt.title("Profit Vs Population")
    plt.xlabel("Population in 10000s")
    plt.ylabel("profit in 100usd")


# In[71]:


plot(X, Y)


# In[97]:


#X=np.stack([np.ones(m), X], axis=1)


# In[18]:


x_mean=np.mean(X)
y_mean=np.mean(Y)


# In[20]:


numerator=0
denominator=0
for i in range(m):
    numerator += (X[i] - x_mean) * (Y[i] - y_mean)
    denominator += (X[i] - x_mean) ** 2
b1 = numerator / denominator
b0 = y_mean - (b1 * x_mean)
print(b1, b0)


# In[63]:


#plotting values 
x_max = np.max(X)
x_min = np.min(X) 
#calculating line values of x and y
x = np.linspace(x_min, x_max)
y = b0 + b1 * x
plt.figure(figsize=(10,10))

#plotting line 
plt.plot(x, y, color='g', label='Linear Regression')
#plot the data point
plt.plot(data[:,0], data[:,1], 'ro', ms=10, mec='k', label='Data Point')
#plt.scatter(X, Y, color='#ff0000', label='Data Point')

# x-axis label
plt.xlabel('Population in 10000s')
#y-axis label
plt.ylabel('profit in 100usd')
plt.legend()
plt.show()


# In[ ]:




