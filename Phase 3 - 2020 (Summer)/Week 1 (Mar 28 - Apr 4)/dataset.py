##import numpy as np
#import pandas as pd

#index = [1, 2, 3, 4, 5, 6, 7]
#a = [np.nan, np.nan, np.nan, 0.1, 0.1, 0.1, 0.1]
#b = [0.2, np.nan, 0.2, 0.2, 0.2, np.nan, np.nan]
#df = pd.DataFrame({'A': a, 'B': b, 'C': c}, index=index)
#df = df.rename_axis('ID')
#print df

import numpy as np
from numpy import loadtxt

#f=open('data.txt','r')
data= np.loadtxt("data.txt",skiprows=1)
#data_array=float (np.loadtxt(f))
print data