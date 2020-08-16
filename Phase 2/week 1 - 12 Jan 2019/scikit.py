import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from matplotlib import style
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
data = pd.read_csv('data.txt',header = None,delimiter = '\t',skiprows = 1)
a = scale(np.array(data))
model = PCA(n_components=10)
model.fit(a)
print(model.explainded_variance_ratio_)
