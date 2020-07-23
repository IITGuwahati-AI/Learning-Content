import numpy as np

a = np.loadtxt(fname = "data.txt" #Specifying the PATH of our .txt file
                    , skiprows = 1     #Skipping first row of the dataset
                    , unpack=False
                    , dtype=float
                    )

print(a[0])