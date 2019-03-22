import os
import numpy as np
import re
from matplotlib import pyplot
from scipy import optimize
from scipy.io import loadmat
import utils
grader = utils.Grader()

def processEmail(email_contents, verbose=True):
    # Load Vocabulary
    vocabList = utils.getVocabList()
    #print(vocabList)

    # Init return value
    word_indices = []

    # ========================== Preprocess Email ===========================
    # Find the Headers ( \n\n and remove )
    # Uncomment the following lines if you are working with raw emails with the
    # full headers
    # hdrstart = email_contents.find(chr(10) + chr(10))
    # email_contents = email_contents[hdrstart:]

    # Lower case
    email_contents = email_contents.lower()
    
    # Strip all HTML
    # Looks for any expression that starts with < and ends with > and replace
    # and does not have any < or > in the tag it with a space
    email_contents =re.compile('<[^<>]+>').sub(' ', email_contents)

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
        
        try:
            index = vocabList.index(word)
            word_indices.append(index)
        except ValueError:
            pass

        # =============================================================

    if verbose:
        print('----------------')
        print('Processed email:')
        print('----------------')
        print(' '.join(processed_email))
    return word_indices

def emailFeatures(word_indices):
    # Total number of words in the dictionary
    n = 1899

    # You need to return the following variables correctly.
    x = np.zeros(n)

    # ===================== YOUR CODE HERE ======================

    for index in word_indices:
        x[index] = 1
    
    # ===========================================================
    
    return x

if __name__=='__main__':
    #  To use an SVM to classify emails into Spam v.s. Non-Spam, you first need
    #  to convert each email into a vector of features. In this part, you will
    #  implement the preprocessing steps for each email. You should
    #  complete the code in processEmail.m to produce a word indices vector
    #  for a given email.
    
    # Extract Features
    with open(os.path.join('Data', 'vocab.txt')) as fid:
        file_contents = fid.read()
        #print(file_contents)
    with open(os.path.join('Data', 'emailSample1.txt')) as fid:
        file_contents = fid.read()
    
    word_indices  = processEmail(file_contents)
    
    #Print Stats
    print('-------------')
    print('Word Indices:')
    print('-------------')
    print(word_indices)
    
    
    # Extract Features
    with open(os.path.join('Data', 'emailSample1.txt')) as fid:
        file_contents = fid.read()
    
    word_indices  = processEmail(file_contents)
    features      = emailFeatures(word_indices)
    
    # Print Stats
    print('\nLength of feature vector: %d' % len(features))
    print('Number of non-zero entries: %d' % sum(features > 0))
    
    
    
    # Load the Spam Email dataset
    # You will have X, y in your environment
    data = loadmat(os.path.join('Data', 'spamTrain.mat'))
    X, y= data['X'].astype(float), data['y'][:, 0]
    
    print('Training Linear SVM (Spam Classification)')
    print('This may take 1 to 2 minutes ...\n')
    
    C = 0.1
    model = utils.svmTrain(X, y, C, utils.linearKernel)
            
    # Compute the training accuracy
    p = utils.svmPredict(model, X)
    
    print('Training Accuracy: %.2f' % (np.mean(p == y) * 100))
    
    # Load the test dataset
    # You will have Xtest, ytest in your environment
    data = loadmat(os.path.join('Data', 'spamTest.mat'))
    Xtest, ytest = data['Xtest'].astype(float), data['ytest'][:, 0]
    
    print('Evaluating the trained Linear SVM on a test set ...')
    p = utils.svmPredict(model, Xtest)
    
    print('Test Accuracy: %.2f' % (np.mean(p == ytest) * 100))
        
    
    # Sort the weights and obtin the vocabulary list
    # NOTE some words have the same weights, 
    # so their order might be different than in the text above
    idx = np.argsort(model['w'])
    top_idx = idx[-15:][::-1]
    vocabList = utils.getVocabList()
    
    print('Top predictors of spam:')
    print('%-15s %-15s' % ('word', 'weight'))
    print('----' + ' '*12 + '------')
    for word, w in zip(np.array(vocabList)[top_idx], model['w'][top_idx]):
        print('%-15s %0.2f' % (word, w))

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    