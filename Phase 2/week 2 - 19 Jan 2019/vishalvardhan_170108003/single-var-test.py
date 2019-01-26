from learning import * 

data = np.loadtxt('2.txt',delimiter = ',')

print("This program doesn't handle errors so pls do input the values in the shown format only.")
print("1.plotdata\n2.comp_cost\n3.step_descent\n4.final_descent")
n = int(input('choose one to test: '))
if n == 1:
	print("\n\n\nIn testing plotdata")
	plot_func(data[:,0], data[:,1], mode = 'scatter', x_label = 'population', y_label = 'profit')
	plt.show()

elif n == 2:
	print('\n\n\nIn testing comp_cost')
	theta = list(map(float, input('theta(ex : 1.2 2.5): ').split()))
	print(comp_cost(data[:,0], data[:,1], theta))

elif n == 3:
	print("\n\n\nIn testing step_descent")
	alpha = float(input('alpha(real number): '))
	theta = list(map(float, input('theta(ex : 1.2 2.5): ').split()))
	plot_func(data[:,0], np.array(list(map(lambda i:theta[0] + theta[1]*i, data[:,0]))), mode = 'plot', color = 'b', label = 'bef-des')	
	theta = step_grad(data[:,0], data[:,1], theta, alpha)
	print(f"theta after one descent : {theta}")
	plot_func(data[:,0], data[:,1], mode = 'scatter', x_label = 'population', y_label = 'profit', color = 'g')
	plot_func(data[:,0], np.array(list(map(lambda i:theta[0] + theta[1]*i, data[:,0]))), mode = 'plot', color = 'r', label = 'after-des')
	plt.show()

elif n == 4:
	print('\n\n\nIn testing final_descent')
	alpha = float(input('alpha(real number): '))
	theta = list(map(float, input('theta(ex : 1.2 2.5): ').split()))
	plot_func(data[:,0], np.array(list(map(lambda i:theta[0] + theta[1]*i, data[:,0]))), mode = 'plot', color = 'b', label = 'bef-des')	
	theta = final_des(data[:,0], data[:,1], theta, alpha)
	print(f"theta after final descent : {theta}")
	plot_func(data[:,0], data[:,1], mode = 'scatter', x_label = 'population', y_label = 'profit', color = 'g')
	plot_func(data[:,0], np.array(list(map(lambda i:theta[0] + theta[1]*i, data[:,0]))), mode = 'plot', color = 'r', label = 'after-des')
	plt.show()

else:
	print('Nothing to show here.try a different value.')	