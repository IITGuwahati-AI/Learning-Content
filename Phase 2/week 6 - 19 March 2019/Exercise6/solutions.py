# used for manipulating directory paths
import os

# Scientific and vector computation for python
import numpy as np

# Import regular expressions to process emails
import re

# Plotting library
from matplotlib import pyplot

# Optimization module in scipy
from scipy import optimize

# will be used to load MATLAB mat datafile format
from scipy.io import loadmat

# library written for this exercise providing additional functions for assignment submission, and others
import utils

# define the submission/grader object for this exercise
grader = utils.Grader()


# Load from ex6data1
# You will have X, y as keys in the dict data
data = loadmat(os.path.join('Data', 'ex6data1.mat'))
X, y = data['X'], data['y'][:, 0]

# Plot training data
utils.plotData(X, y)
pyplot.show()

# You should try to change the C value below and see how the decision
# boundary varies (e.g., try C = 1000)
C = 100

model = utils.svmTrain(X, y, C, utils.linearKernel, 1e-3, 20)
utils.visualizeBoundaryLinear(X, y, model)
pyplot.show()



def gaussianKernel(x1, x2, sigma):
    """
    Computes the radial basis function
    Returns a radial basis function kernel between x1 and x2.
    
    Parameters
    ----------
    x1 :  numpy ndarray
        A vector of size (n, ), representing the first datapoint.
    
    x2 : numpy ndarray
        A vector of size (n, ), representing the second datapoint.
    
    sigma : float
        The bandwidth parameter for the Gaussian kernel.

    Returns
    -------
    sim : float
        The computed RBF between the two provided data points.
    
    Instructions
    ------------
    Fill in this function to return the similarity between `x1` and `x2`
    computed using a Gaussian kernel with bandwidth `sigma`.
    """
    sim = 0
    # ====================== YOUR CODE HERE ======================
    sim = np.exp(-np.dot(x1-x2,x1-x2)/(2*sigma*sigma))


    # =============================================================
    return sim

x1 = np.array([1, 2, 1])
x2 = np.array([0, 4, -1])
sigma = 2

sim = gaussianKernel(x1, x2, sigma)

print('Gaussian Kernel between x1 = [1, 2, 1], x2 = [0, 4, -1], sigma = %0.2f:'
	'\n\t%f\n(for sigma = 2, this value should be about 0.324652)\n' % (sigma, sim))



# grader[1] = gaussianKernel
# grader.grade()



# Load from ex6data2
# You will have X, y as keys in the dict data
data = loadmat(os.path.join('Data', 'ex6data2.mat'))
X, y = data['X'], data['y'][:, 0]

# Plot training data
utils.plotData(X, y)
pyplot.show()



# SVM Parameters
C = 1
sigma = 0.1

model= utils.svmTrain(X, y, C, gaussianKernel, args=(sigma,))
utils.visualizeBoundary(X, y, model)
pyplot.show()



# Load from ex6data3
# You will have X, y, Xval, yval as keys in the dict data
data = loadmat(os.path.join('Data', 'ex6data3.mat'))
X, y, Xval, yval = data['X'], data['y'][:, 0], data['Xval'], data['yval'][:, 0]

# Plot training data
utils.plotData(X, y)
pyplot.show()


def dataset3Params(X, y, Xval, yval):
    """
    Returns your choice of C and sigma for Part 3 of the exercise 
    where you select the optimal (C, sigma) learning parameters to use for SVM
    with RBF kernel.
    
    Parameters
    ----------
    X : array_like
        (m x n) matrix of training data where m is number of training examples, and 
        n is the number of features.
    
    y : array_like
        (m, ) vector of labels for ther training data.
    
    Xval : array_like
        (mv x n) matrix of validation data where mv is the number of validation examples
        and n is the number of features
    
    yval : array_like
        (mv, ) vector of labels for the validation data.
    
    Returns
    -------
    C, sigma : float, float
        The best performing values for the regularization parameter C and 
        RBF parameter sigma.
    
    Instructions
    ------------
    Fill in this function to return the optimal C and sigma learning 
    parameters found using the cross validation set.
    You can use `svmPredict` to predict the labels on the cross
    validation set. For example, 
    
        predictions = svmPredict(model, Xval)

    will return the predictions on the cross validation set.
    
    Note
    ----
    You can compute the prediction error using 
    
        np.mean(predictions != yval)
        """
    # You need to return the following variables correctly.
    C = 1
    sigma = 0.3

    # ====================== YOUR CODE HERE ======================
    values = (0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30)
    lis = []
    for i in values:
    	for j in values:
    		model= utils.svmTrain(X, y, i, gaussianKernel, args=(j,))		
    		predictions = utils.svmPredict(model, Xval)
    		lis += [[i,j,np.mean(predictions != yval)]]
    lis =np.array(lis)
    index = (np.argmin(lis[:,2]))
    C = lis[index,0]
    sigma = lis[index,1]
    # ============================================================
    return C, sigma


# Try different SVM Parameters here
C, sigma = dataset3Params(X, y, Xval, yval)

# Train the SVM
# model = utils.svmTrain(X, y, C, lambda x1, x2: gaussianKernel(x1, x2, sigma))
model = utils.svmTrain(X, y, C, gaussianKernel, args=(sigma,))
utils.visualizeBoundary(X, y, model)
pyplot.show()
print(C, sigma)



# grader[2] = lambda : (C, sigma)
# grader.grade()



def processEmail(email_contents,verbose=True):
	vocabList = utils.getVocabList()
	# print(vocabList)
	word_indices = []
	email_contents = email_contents.lower()
	email_contents = re.compile('<[^<>]+>').sub(' ', email_contents)
	# Handle Numbers
	# Look for one or more characters between 0-9
	email_contents = re.compile('[0-9]+').sub(' number ', email_contents)
	# Handle URLS
	# Look for strings starting with http:// or https://
	email_contents = re.compile('(http|https)://[^\s]*').sub(' httpaddr ', email_contents)
	# Handle Email Addresses
	# Look for strings with @ in the middle
	email_contents = re.compile('[^\s]+@[^\s]+').sub(' emailaddr ', email_contents)
	# Handle $ sign
	email_contents = re.compile('[$]+').sub(' dollar ', email_contents)
	# get rid of any punctuation
	email_contents = re.split('[ @$/#.-:&*+=\[\]?!(){},''">_<;%\n\r]', email_contents)
	# remove any empty word string
	email_contents = [word for word in email_contents if len(word) > 0]
	# Stem the email contents word by word
	stemmer = utils.PorterStemmer()
	processed_email = []
	for word in email_contents:
		# Remove any remaining non alphanumeric characters in word
		word = re.compile('[^a-zA-Z0-9]').sub('', word).strip()
		word = stemmer.stem(word)
		processed_email.append(word)

		if len(word) < 1:
			continue


	# Look up the word in the dictionary and add to word_indices if found
	# ====================== YOUR CODE HERE ======================
	for word in processed_email:
		if word in vocabList:
			word_indices.append(vocabList.index(word)+1)

	# =============================================================
	if verbose:
		print('----------------')
		print('Processed email:')
		print('----------------')
		print(' '.join(processed_email))
		return word_indices

#  To use an SVM to classify emails into Spam v.s. Non-Spam, you first need
#  to convert each email into a vector of features. In this part, you will
#  implement the preprocessing steps for each email. You should
#  complete the code in processEmail.m to produce a word indices vector
#  for a given email.

# Extract Features
with open(os.path.join('Data', 'emailSample1.txt')) as fid:
	file_contents = fid.read()

	word_indices  = processEmail(file_contents)

#Print Stats
print('-------------')
print('Word Indices:')
print('-------------')
print(word_indices)



