import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import style
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
style.use('ggplot')

def read_data():
    df = pd.read_html('https://github.com/IITGuwahati-AI/Learning-Content/blob/master/Phase%203%20-%202020%20(Summer)/Week%201%20(Mar%2028%20-%20Apr%204)/assignment/data.txt')
    return df

def columns(df):
    return df[0][1][0].split("\t")

def fill_data(data, df):
    print('Filling data ....')
    for i in range(11):
        column = columns(df)
        arr = []
        for j in range(1, 1000):
            arr.append(df[0][1][j].split("\t")[i])
        data[column[i]] = arr
    print("Done Filling Data !!")

def plot_graph(data, i, j):
    arr_1_x = []
    arr_1_y = []
    arr_2_x = []
    arr_2_y = []
    for k in range(999):
        if data.T[0][k] == 1.0:
            #plt.scatter(k, data.T[j][k], color='r')
            arr_1_x.append(data.T[i][k])
            arr_1_y.append(data.T[j][k])

        elif data.T[0][k] == 2.0:
            #plt.scatter(k, data.T[j][k], color='b')
            arr_2_x.append(data.T[i][k])
            arr_2_y.append(data.T[j][k])

    plt.xlabel("feature {}".format(i))
    plt.ylabel("feature {}".format(j))
    plt.title("feature {} vs feature {}".format(j, i))
    plt.scatter(arr_1_x, arr_1_y, color='r', label='label 1')
    plt.scatter(arr_2_x, arr_2_y, color='b', label='label 2')
    plt.legend()
    plt.show()
    #plt.savefig('/home/samarth/ML_Tutorials/Week1/Samarth_180103064/Graphs/{} vs {}.png'.format(j, i))

def find_best_features_PCA(data):
    overall_max = 0
    for i in range(1, 11):
        for j in range(i+1, 11):
            features = []
            features.append(str(i))
            features.append(str(j))

            # Separating out the features
            x = data.loc[:, features].values

            # Standardizing the features
            x = StandardScaler().fit_transform(x)

            pca = PCA(n_components=1)
            principalComponents = pca.fit_transform(x)
            max_i_j = 0
            for k in range(999):
                if principalComponents.T[0][k] > max_i_j:
                    max_i_j = principalComponents.T[0][k]
            if max_i_j > overall_max:
                overall_max = max_i_j
                best_features = (i, j)

    return best_features

#extracting the data and saving it
#df = read_data()
#data = pd.DataFrame()
#fill_data(data, df)
#data.to_pickle('data.pickle')

#reading data into numpy array
data = pd.read_pickle('data.pickle')
data = np.array(data, dtype=np.float64)

#drawing plots
plot_graph(data, 1, 2)
print("According to the graphs...")
print("The best features are 1 and 2..")
print("As both the labels can be perfectly separated \n")

#using scikit-learn to apply PCA to find best 2 features
data = pd.read_pickle('data.pickle')
best_features = find_best_features_PCA(data)
print('The best Features according to PCA are : {}'.format(best_features))
