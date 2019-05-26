import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import style
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
data = pd.read_csv('data.txt', header = None,skiprows=1,delimiter='\t')
df = np.array(data)
df = scale(df)
my_model = PCA(n_components=10)
my_model.fit(df)
print(my_model.explained_variance_ratio_)
