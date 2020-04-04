import numpy as np
import matplotlib.pyplot as plt
#loading data using numpy.data variable is a numpy array)
data=np.loadtxt('data.txt',delimiter='\t',skiprows=1)
print(data.shape)
print(type(data))
x,lbl=data[:,1:],data[:,0]
#plotting data and saving all the plots into a single pdf
import matplotlib.backends.backend_pdf
pdf = matplotlib.backends.backend_pdf.PdfPages("output.pdf")
def plotdata(x,y,fx,fy):
    one=y==1
    two=y==2
    fig=plt.figure()
    #fig.savefig(filename)
    plt.plot(x[one,fx],x[one,fy],'ro')
    plt.plot(x[two,fx],x[two,fy],'b*')
    plt.xlabel('feature '+str(fx+1))
    plt.ylabel('feature '+str(fy+1))
    plt.legend(['one','two'])
    plt.title("Feature "+str(fy+1)+" vs Feature "+str(fx+1)+" for Labels 1 and 2")
    plt.show()
    pdf.savefig(fig)
for i in range(0,9):
    for j in range(i+1,10):
        plotdata(x,lbl,i,j)
pdf.close()
print("From the plots it can be seen that features 1 and 2 best classify the labels" )

#using PCA    
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
x=StandardScaler().fit_transform(x)
pca=PCA(n_components=5).fit(x)
new=pca.transform(x)
print(pca.explained_variance_ratio_)#principal components variance values are printed
print(abs(pca.components_))#prints the variance value for all the features
