# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 14:54:41 2020

@author: LENOVO
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 19:13:26 2020

@author: LENOVO
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
plt.style.use('ggplot')


ds = pd.read_csv(r"C:\Users\LENOVO\Desktop\data.csv", delimiter='\t')
ds1=ds.loc[ds['Label']==1]
ds2=ds.loc[ds['Label']==2]


plt.rcParams['figure.figsize']=(30,30)
fig, axes=plt.subplots(10,10)




for i in range(1,11):
    for j in range(1,11):
        if(i!=j):
            x=str(i)
            y=str(j)
            ds1=ds.loc[ds['Label']==1].sort_values(by=x)
            ds2=ds.loc[ds['Label']==2].sort_values(by=x)
            x_c1=ds1[[x]]
            y_c1=ds1[[y]]
            x_c2=ds2[[x]]
            y_c2=ds2[[y]]
            axes[i-1][j-1].plot(x_c1,y_c1,'r',label='1')
            axes[i-1][j-1].plot(x_c2,y_c2,'b',label='2') 
            axes[i-1][j-1].set_xlabel(x)
            axes[i-1][j-1].set_ylabel(y)

        
fig.tight_layout(pad=2.0)
plt.show()