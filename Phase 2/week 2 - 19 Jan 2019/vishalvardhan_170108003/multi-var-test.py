from learning import *

data = np.loadtxt('3.txt',delimiter = ',')

print("1.feature Normalise\n2.comp_cost\n3.step_descent\n4.final_descent\n5.normal equation")
n = int(input('choose one to test: '))

if n == 1:
	print('\n\n\nIn testing feature Normalise.')
	data[:,[0,1]] = feat_normalise(data[:,[0,1]])
	print(data)

elif n == 2:
	print('\n\n\nIn testing comp_cost')
	theta = list(map(float, input('theta(ex : 1.2 2.5 0): ').split()))
	print(comp_cost(data[:,[0,1]], data[:,2], theta))

elif n == 3:
	print('\n\n\nIn testing step_descent')
	theta = list(map(float, input('theta(ex : 1.2 2.5 0): ').split()))
	alpha = float(input('alpha (ex: 0.01): '))
	theta = np.array(theta)
	theta = step_grad(data[:,[0,1]],data[:,2],theta,alpha)
	print(theta)

elif n == 4:
	print('\n\n\nIn testing final_descent')
	alpha = float(input('alpha(real number): '))
	theta = list(map(float, input('theta(ex : 1.2 2.5 0.2): ').split()))
	# data = feat_normalise(data)
	data[:,0] = data[:,0]/1000
	data[:,2] = data[:,2]/100000
	theta = final_des(data[:,[0,1]], data[:,2], theta, alpha)
	theta[0] *= 100000
	theta[1] *= 100
	theta[2] *= 100000
	print(f"theta after final descent : {theta}")
	# print(f"cost after final descent : {comp_cost(data[:,[0,1]], data[:,2], theta)}")

elif n == 5:
	print('\n\n\nIn testing normal equation.')
	theta = normal_equation(data[:,[0,1]], data[:,2])
	print(f"theta found is : {theta}")
	print(f"minim cost is : {comp_cost(data[:,[0,1]], data[:,2], theta)}")
else:
	print("nothing's up here try again")