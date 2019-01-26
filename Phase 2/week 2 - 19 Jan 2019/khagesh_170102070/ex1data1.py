import numpy as np;
import matplotlib.pyplot as plt;

def gradient_descent(x,y,alpha,b,m):
    count=0;number=list();jo=list();
    for i in range(10000):
        h=np.dot(x,b);
        b=b-(1/m)*alpha*np.sum(np.dot(x.T,h-y));
        j=1/(2*m)*np.sum((h-y)**2);
        count+=1; jo.append(j); number.append(count);
    return b,number,jo;

def plot_J(iterations,J):
    iterations=iterations[0:1000];J=J[0:1000];
    plt.plot(iterations,J,color='g');
    plt.xlabel('count');
    plt.ylabel('J');
    plt.title('CostFunction vs number of iterations');
    plt.show();
    
def linearregression(x,y,b):
    X=x.T;
    plt.scatter(X[1],y,color="r",marker="o",s=10);
    y=np.dot(x,b);
    print('y =',b[0],'+',b[1],'x');
    plt.title('Profit vs Population');
    x=x.T;y=y.T;
    plt.plot(x[1],y,color="b");
    plt.xlabel('Population');
    plt.ylabel('Profit');
    plt.show();
    return y;

def preference(Y,x,y):
    diff=y-Y;
    d = dict(enumerate(diff.flatten(), 0))
    d = {v: k for k, v in d.items()}
    sorted_profit=sorted(d,reverse=True);
    for i in sorted_profit:
        index = d[i];
        print('City number',index+1,'with Population',x[index],', profit',y[index]);

if __name__=="__main__":
    data=np.loadtxt("ex1data1.txt",delimiter=',').T;
    x=np.array([np.ones(np.size(data[0])),data[0]]);
    y=data[1];
    x=x.T; y=y.T; alpha=0.00003; m,n=x.shape; b=np.array([0,1]).T;
    b,iterations,J = gradient_descent(x,y,alpha,b,m);
    
    plot_J(iterations,J);
    Y=linearregression(x,y,b);
    
    preference(Y,x.T[1],y);
    
    
