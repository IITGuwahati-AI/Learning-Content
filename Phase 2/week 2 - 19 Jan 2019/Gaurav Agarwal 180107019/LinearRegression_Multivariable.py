#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from matplotlib import pyplot as plt


# In[65]:


data=np.loadtxt("./ex1data2.txt", delimiter=",")
print(data)
X=data[:,0:2]
m=data[:,0].size
print(m)


# In[59]:


print(X)
print(X.shape)


# In[60]:


Y2=np.ones(47)
print(Y2)


# In[82]:



X = np.insert(X, obj=0, values=Y2, axis=1)
print(X)


# In[83]:


Y=data[:,2]
print(Y)
print(X)


# In[87]:


arr=[]
j=np.matmul(X.transpose(), X)
new=np.linalg.inv(j)
new2=np.matmul(new, X.transpose())
arr=np.matmul(new2, Y)
print(arr)


# In[88]:


print(arr)


# In[92]:





# In[ ]:





# In[ ]:




