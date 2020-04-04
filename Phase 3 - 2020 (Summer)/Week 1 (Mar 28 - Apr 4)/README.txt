>>From the graphs, it is quite clear that the features 1 and 2 best classify the labels...\

>>Using PCA:
	>>Here we observe that the variance ratio for each of the features respectively in its original form are:
		[0.19550175 0.17821025 0.1740652  0.10549293 0.10017266 0.09534421 0.07929688 0.04445225 0.02060953 0.00685434]
		
	
	>>This clearly indicates the highest variance of feature 1 and feature 2 and hence the previous result 
          gained from the graphs.
	






	>>Now, applying PCA over two corresponding features at a time for all of them and projecting onto a single component,we get these:
		1 2 [0.60185502]    1 3 [0.51331434]     1 4 [0.53577351]    1 5 [0.5122996]
		1 6 [0.51136194]    1 7 [0.52795251]     1 8 [0.5080902]     1 9 [0.50551827]
		1 10 [0.70843126]   2 3 [0.51284582]     2 4 [0.51364179]    2 5 [0.51399512]
		2 6 [0.50957615]    2 7 [0.52942058]     2 8 [0.51392033     2 9 [0.50525977]
		2 10 [0.74562299]   3 4 [0.52596273]     3 5 [0.5116986]     3 6 [0.50144882]
		3 7 [0.51624331]    3 8 [0.96538812]     3 9 [0.50718756]    3 10 [0.50399196]
		4 5 [0.51243867]    4 6 [0.50208561]     4 7 [0.51966986]    4 8 [0.51660687]
		4 9 [0.51070635]    4 10 [0.51570476]    5 6 [0.52352081]    5 7 [0.51257143]
		5 8 [0.51345915]    5 9 [0.71532175]     5 10 [0.50233362]   6 7 [0.50159157]
		6 8 [0.50274626]    6 9 [0.50441627]     6 10 [0.50571473]   7 8 [0.50919461]
		7 9 [0.82504383]    7 10 [0.51820473]    8 9 [0.50661449]    8 10 [0.50141312]
		9 10 [0.5031281]
	>>Here, we observe that the highest variance comes from the features 3 and 8 which is nowhere near to the results from the graph.
	>>This is because the data has been transformed due to change of the components similar to change of axes and this observation 
	  cannot be traced to the corresponding original data. Hence, the data observed.	 