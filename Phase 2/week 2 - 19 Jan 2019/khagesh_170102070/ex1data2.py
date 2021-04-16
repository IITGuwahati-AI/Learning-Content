import numpy as np;

if __name__=="__main__":
    data=np.loadtxt("ex1data2.txt",delimiter=',');
    data=data.T;
    x=data[0];
    z=data[2]
    
    X=np.array([np.ones(np.size(x)),x,data[1]]).T;
   
    Y=np.array([z]).T
    
    theta=np.linalg.inv(np.dot(X.T,X));
    theta=np.dot(theta,X.T);
    theta=np.dot(theta,Y);
    
    new_Y=np.dot(X,theta);
    
    max_price_diff = z-new_Y.T;
    max_price=np.amax(max_price_diff);
    s=np.size(data[0]);
    index=0;
    for i in range(s):
        if max_price_diff[0][i] == max_price:
            index=i;
    
    print('Good market price ')

    print('price = ({:2f})+({:2f})size+({:2f})bedrooms'.format(theta[0][0],theta[1][0],theta[2][0]));
    
    #print(data[2][index]);
    print('max deviation:',end="");
    print(theta[0][0]+theta[1][0]*data[0][index]+theta[2][0]*data[1][index]);
    
    
    
