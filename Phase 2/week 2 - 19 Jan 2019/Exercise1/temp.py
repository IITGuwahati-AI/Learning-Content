import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

data = np.loadtxt("/home/abhishek/ex1data1.txt", delimiter=',')
m=len(data)
p=np.size(data,1)
alpha = 0.01
iterations = 1500
theta = np.zeros((p,1))
y = data[:,-1:]


X = np.ones((m,p))
X[:,1:] = data[:,0:p-1]

theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)
Jvals = np.zeros((np.size(theta0_vals),np.size(theta1_vals)))

for i in range(0,len(theta0_vals)):
   for j in range(0, len(theta1_vals)):
         theta_val = np.array([i,j])
         theta_val = theta_val.T
         cost = np.dot(X,theta_val) - y
         cost = cost*cost
         sq= np.sum(cost)
         Jvals[i,j] = sq/(2*m)

Jvals = Jvals.T
plt.contour(theta0_vals, theta1_vals, Jvals)
plt.xlabel('theta_0')

plt.savefig("Contour Plot of J vs theta_0 and theta_1")
plt.show()

