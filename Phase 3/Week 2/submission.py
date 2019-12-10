from urllib.parse import urlencode
from urllib.request import urlopen
import pickle
import json
from collections import OrderedDict
import numpy as np
import os


class SubmissionBase:

    submit_url = 'https://www-origin.coursera.org/api/' \
                 'onDemandProgrammingImmediateFormSubmissions.v1'
    save_file = 'token.pkl'

    def __init__(self, assignment_slug, part_names):
        self.assignment_slug = assignment_slug
        self.part_names = part_names
        self.login = None
        self.token = None
        self.functions = OrderedDict()
        self.args = dict()

    def grade(self):
        print('\nSubmitting Solutions | Programming Exercise %s\n' % self.assignment_slug)
        self.login_prompt()

        # Evaluate the different parts of exercise
        parts = OrderedDict()
        for part_id, result in self:
            parts[str(part_id)] = {'output': sprintf('%0.5f ', result)}
        result, response = self.request(parts)
        response = json.loads(response.decode("utf-8"))

        # if an error was returned, print it and stop
        if 'errorMessage' in response:
            print(response['errorMessage'])
            return

        # Print the grading table
        print('%43s | %9s | %-s' % ('Part Name', 'Score', 'Feedback'))
        print('%43s | %9s | %-s' % ('---------', '-----', '--------'))
        for part in parts:
            part_feedback = response['partFeedbacks'][part]
            part_evaluation = response['partEvaluations'][part]
            score = '%d / %3d' % (part_evaluation['score'], part_evaluation['maxScore'])
            print('%43s | %9s | %-s' % (self.part_names[int(part) - 1], score, part_feedback))
        evaluation = response['evaluation']
        total_score = '%d / %d' % (evaluation['score'], evaluation['maxScore'])
        print('                                  --------------------------------')
        print('%43s | %9s | %-s\n' % (' ', total_score, ' '))

    def login_prompt(self):
        if os.path.isfile(self.save_file):
            with open(self.save_file, 'rb') as f:
                login, token = pickle.load(f)
            reenter = input('Use token from last successful submission (%s)? (Y/n): ' % login)

            if reenter == '' or reenter[0] == 'Y' or reenter[0] == 'y':
                self.login, self.token = login, token
                return
            else:
                os.remove(self.save_file)

        self.login = input('Login (email address): ')
        self.token = input('Token: ')

        # Save the entered credentials
        if not os.path.isfile(self.save_file):
            with open(self.save_file, 'wb') as f:
                pickle.dump((self.login, self.token), f)

    def request(self, parts):
        params = {
            'assignmentSlug': self.assignment_slug,
            'secret': self.token,
            'parts': parts,
            'submitterEmail': self.login}

        params = urlencode({'jsonBody': json.dumps(params)}).encode("utf-8")
        f = urlopen(self.submit_url, params)
        try:
            return 0, f.read()
        finally:
            f.close()

    def __iter__(self):
        for part_id in self.functions:
            yield part_id

    def __setitem__(self, key, value):
        self.functions[key] = value


def sprintf(fmt, arg):
    """ Emulates (part of) Octave sprintf function. """
    if isinstance(arg, tuple):
        # for multiple return values, only use the first one
        arg = arg[0]

    if isinstance(arg, (np.ndarray, list)):
        # concatenates all elements, column by column
        return ' '.join(fmt % e for e in np.asarray(arg).ravel('F'))
    else:
        return fmt % arg
    {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Programming Exercise 1: Linear Regression\n",
    "\n",
    "## Introduction\n",
    "\n",
    "In this exercise, you will implement linear regression and get to see it work on data. Before starting on this programming exercise, we strongly recommend watching the video lectures and completing the review questions for the associated topics.\n",
    "\n",
    "All the information you need for solving this assignment is in this notebook, and all the code you will be implementing will take place within this notebook. The assignment can be promptly submitted to the coursera grader directly from this notebook (code and instructions are included below).\n",
    "\n",
    "Before we begin with the exercises, we need to import all libraries required for this programming exercise. Throughout the course, we will be using [`numpy`](http://www.numpy.org/) for all arrays and matrix operations, and [`matplotlib`](https://matplotlib.org/) for plotting.\n",
    "\n",
    "You can find instructions on how to install required libraries in the README file in the [github repository](https://github.com/dibgerge/ml-coursera-python-assignments)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "# used for manipulating directory paths\n",
    "import os\n",
    "\n",
    "# Scientific and vector computation for python\n",
    "import numpy as np\n",
    "\n",
    "# Plotting library\n",
    "from matplotlib import pyplot\n",
    "from mpl_toolkits.mplot3d import Axes3D  # needed to plot 3-D surfaces\n",
    "\n",
    "# library written for this exercise providing additional functions for assignment submission, and others\n",
    "import utils \n",
    "\n",
    "# define the submission/grader object for this exercise\n",
    "grader = utils.Grader()\n",
    "\n",
    "# tells matplotlib to embed plots within the notebook\n",
    "%matplotlib inline"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Submission and Grading\n",
    "\n",
    "After completing each part of the assignment, be sure to submit your solutions to the grader.\n",
    "\n",
    "For this programming exercise, you are only required to complete the first part of the exercise to implement linear regression with one variable. The second part of the exercise, which is optional, covers linear regression with multiple variables. The following is a breakdown of how each part of this exercise is scored.\n",
    "\n",
    "**Required Exercises**\n",
    "\n",
    "| Section | Part                                           |Submitted Function                     | Points \n",
    "|---------|:-                                             |:-                                     | :-:    \n",
    "| 1       | [Warm up exercise](#section1)                  | [`warmUpExercise`](#warmUpExercise)    |  10    \n",
    "| 2       | [Compute cost for one variable](#section2)     | [`computeCost`](#computeCost)         |  20    \n",
    "| 3       | [Gradient descent for one variable](#section3) | [`gradientDescent`](#gradientDescent) |  20    \n",
    "| 4       | [Feature normalization](#section4)                   | [`featureNormalize`](#featureNormalize) | 10      |\n",
    "| 5       | [Compute cost for multiple variables](#section5)     | [`computeCostMulti`](#computeCostMulti) | 20      |\n",
    "| 6       | [Gradient descent for multiple variables](#section5) | [`gradientDescentMulti`](#gradientDescentMulti) |10      |\n",
    "| 7       | [Normal Equations](#section7)                        | [`normalEqn`](#normalEqn)        | 10      |\n",
    "|         | Total Points                                   |                                       | 100    \n",
    "\n",
    "You are allowed to submit your solutions multiple times, and we will take only the highest score into consideration.\n",
    "\n",
    "<div class=\"alert alert-block alert-warning\">\n",
    "At the end of each section in this notebook, we have a cell which contains code for submitting the solutions thus far to the grader. Execute the cell to see your score up to the current section. For all your work to be submitted properly, you must execute those cells at least once. They must also be re-executed everytime the submitted function is updated.\n",
    "</div>\n",
    "\n",
    "\n",
    "## Debugging\n",
    "\n",
    "Here are some things to keep in mind throughout this exercise:\n",
    "\n",
    "- Python array indices start from zero, not one (contrary to OCTAVE/MATLAB). \n",
    "\n",
    "- There is an important distinction between python arrays (called `list` or `tuple`) and `numpy` arrays. You should use `numpy` arrays in all your computations. Vector/matrix operations work only with `numpy` arrays. Python lists do not support vector operations (you need to use for loops).\n",
    "\n",
    "- If you are seeing many errors at runtime, inspect your matrix operations to make sure that you are adding and multiplying matrices of compatible dimensions. Printing the dimensions of `numpy` arrays using the `shape` property will help you debug.\n",
    "\n",
    "- By default, `numpy` interprets math operators to be element-wise operators. If you want to do matrix multiplication, you need to use the `dot` function in `numpy`. For, example if `A` and `B` are two `numpy` matrices, then the matrix operation AB is `np.dot(A, B)`. Note that for 2-dimensional matrices or vectors (1-dimensional), this is also equivalent to `A@B` (requires python >= 3.5)."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<a id=\"section1\"></a>\n",
    "## 1 Simple python and `numpy` function\n",
    "\n",
    "The first part of this assignment gives you practice with python and `numpy` syntax and the homework submission process. In the next cell, you will find the outline of a `python` function. Modify it to return a 5 x 5 identity matrix by filling in the following code:\n",
    "\n",
    "```python\n",
    "A = np.eye(5)\n",
    "```\n",
    "<a id=\"warmUpExercise\"></a>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "def warmUpExercise():\n",
    "    \"\"\"\n",
    "    Example function in Python which computes the identity matrix.\n",
    "    \n",
    "    Returns\n",
    "    -------\n",
    "    A : array_like\n",
    "        The 5x5 identity matrix.\n",
    "    \n",
    "    Instructions\n",
    "    ------------\n",
    "    Return the 5x5 identity matrix.\n",
    "    \"\"\"    \n",
    "    # ======== YOUR CODE HERE ======\n",
    "    A = []   # modify this line\n",
    "    A = np.eye(5)\n",
    "    # ==============================\n",
    "    return A"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The previous cell only defines the function `warmUpExercise`. We can now run it by executing the following cell to see its output. You should see output similar to the following:\n",
    "\n",
    "```python\n",
    "array([[ 1.,  0.,  0.,  0.,  0.],\n",
    "       [ 0.,  1.,  0.,  0.,  0.],\n",
    "       [ 0.,  0.,  1.,  0.,  0.],\n",
    "       [ 0.,  0.,  0.,  1.,  0.],\n",
    "       [ 0.,  0.,  0.,  0.,  1.]])\n",
    "```"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[1., 0., 0., 0., 0.],\n",
       "       [0., 1., 0., 0., 0.],\n",
       "       [0., 0., 1., 0., 0.],\n",
       "       [0., 0., 0., 1., 0.],\n",
       "       [0., 0., 0., 0., 1.]])"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "warmUpExercise()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 1.1 Submitting solutions\n",
    "\n",
    "After completing a part of the exercise, you can submit your solutions for grading by first adding the function you modified to the grader object, and then sending your function to Coursera for grading. \n",
    "\n",
    "The grader will prompt you for your login e-mail and submission token. You can obtain a submission token from the web page for the assignment. You are allowed to submit your solutions multiple times, and we will take only the highest score into consideration.\n",
    "\n",
    "Execute the next cell to grade your solution to the first part of this exercise.\n",
    "\n",
    "*You should now submit your solutions.*"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "Submitting Solutions | Programming Exercise linear-regression\n",
      "\n",
      "Login (email address): k.kaushik25@gmail.com\n",
      "Token: K9kp02dhKpbexGwY\n",
      "                                  Part Name |     Score | Feedback\n",
      "                                  --------- |     ----- | --------\n",
      "                           Warm up exercise |  10 /  10 | Nice work!\n",
      "          Computing Cost (for one variable) |   0 /  40 | \n",
      "        Gradient Descent (for one variable) |   0 /  50 | \n",
      "                      Feature Normalization |   0 /   0 | \n",
      "    Computing Cost (for multiple variables) |   0 /   0 | \n",
      "  Gradient Descent (for multiple variables) |   0 /   0 | \n",
      "                           Normal Equations |   0 /   0 | \n",
      "                                  --------------------------------\n",
      "                                            |  10 / 100 |  \n",
      "\n"
     ]
    }
   ],
   "source": [
    "# appends the implemented function in part 1 to the grader object\n",
    "grader[1] = warmUpExercise\n",
    "\n",
    "# send the added functions to coursera grader for getting a grade on this part\n",
    "grader.grade()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2 Linear regression with one variable\n",
    "\n",
    "Now you will implement linear regression with one variable to predict profits for a food truck. Suppose you are the CEO of a restaurant franchise and are considering different cities for opening a new outlet. The chain already has trucks in various cities and you have data for profits and populations from the cities. You would like to use this data to help you select which city to expand to next. \n",
    "\n",
    "The file `Data/ex1data1.txt` contains the dataset for our linear regression problem. The first column is the population of a city (in 10,000s) and the second column is the profit of a food truck in that city (in $10,000s). A negative value for profit indicates a loss. \n",
    "\n",
    "We provide you with the code needed to load this data. The dataset is loaded from the data file into the variables `x` and `y`:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Read comma separated data\n",
    "data = np.loadtxt(os.path.join('Data', 'ex1data1.txt'), delimiter=',')\n",
    "X, y = data[:, 0], data[:, 1]\n",
    "\n",
    "m = y.size  # number of training examples"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 2.1 Plotting the Data\n",
    "\n",
    "Before starting on any task, it is often useful to understand the data by visualizing it. For this dataset, you can use a scatter plot to visualize the data, since it has only two properties to plot (profit and population). Many other problems that you will encounter in real life are multi-dimensional and cannot be plotted on a 2-d plot. There are many plotting libraries in python (see this [blog post](https://blog.modeanalytics.com/python-data-visualization-libraries/) for a good summary of the most popular ones). \n",
    "\n",
    "In this course, we will be exclusively using `matplotlib` to do all our plotting. `matplotlib` is one of the most popular scientific plotting libraries in python and has extensive tools and functions to make beautiful plots. `pyplot` is a module within `matplotlib` which provides a simplified interface to `matplotlib`'s most common plotting tasks, mimicking MATLAB's plotting interface.\n",
    "\n",
    "<div class=\"alert alert-block alert-warning\">\n",
    "You might have noticed that we have imported the `pyplot` module at the beginning of this exercise using the command `from matplotlib import pyplot`. This is rather uncommon, and if you look at python code elsewhere or in the `matplotlib` tutorials, you will see that the module is named `plt`. This is used by module renaming by using the import command `import matplotlib.pyplot as plt`. We will not using the short name of `pyplot` module in this class exercises, but you should be aware of this deviation from norm.\n",
    "</div>\n",
    "\n",
    "\n",
    "In the following part, your first job is to complete the `plotData` function below. Modify the function and fill in the following code:\n",
    "\n",
    "```python\n",
    "    pyplot.plot(x, y, 'ro', ms=10, mec='k')\n",
    "    pyplot.ylabel('Profit in $10,000')\n",
    "    pyplot.xlabel('Population of City in 10,000s')\n",
    "```"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "def plotData(x, y):\n",
    "    \"\"\"\n",
    "    Plots the data points x and y into a new figure. Plots the data \n",
    "    points and gives the figure axes labels of population and profit.\n",
    "    \n",
    "    Parameters\n",
    "    ----------\n",
    "    x : array_like\n",
    "        Data point values for x-axis.\n",
    "\n",
    "    y : array_like\n",
    "        Data point values for y-axis. Note x and y should have the same size.\n",
    "    \n",
    "    Instructions\n",
    "    ------------\n",
    "    Plot the training data into a figure using the \"figure\" and \"plot\"\n",
    "    functions. Set the axes labels using the \"xlabel\" and \"ylabel\" functions.\n",
    "    Assume the population and revenue data have been passed in as the x\n",
    "    and y arguments of this function.    \n",
    "    \n",
    "    Hint\n",
    "    ----\n",
    "    You can use the 'ro' option with plot to have the markers\n",
    "    appear as red circles. Furthermore, you can make the markers larger by\n",
    "    using plot(..., 'ro', ms=10), where `ms` refers to marker size. You \n",
    "    can also set the marker edge color using the `mec` property.\n",
    "    \"\"\"\n",
    "    fig = pyplot.figure()  # open a new figure\n",
    "    \n",
    "    # ====================== YOUR CODE HERE ======================= \n",
    "    pyplot.plot(x, y, 'ro', ms=10, mec='k')\n",
    "    pyplot.ylabel('Profit in $10,000')\n",
    "    pyplot.xlabel('Population of City in 10,000s')\n",
    "    \n",
    "\n",
    "    # =============================================================\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Now run the defined function with the loaded data to visualize the data. The end result should look like the following figure:\n",
    "\n",
    "![](Figures/dataset1.png)\n",
    "\n",
    "Execute the next cell to visualize the data."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plotData(X, y)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "To quickly learn more about the `matplotlib` plot function and what arguments you can provide to it, you can type `?pyplot.plot` in a cell within the jupyter notebook. This opens a separate page showing the documentation for the requested function. You can also search online for plotting documentation. \n",
    "\n",
    "To set the markers to red circles, we used the option `'or'` within the `plot` function."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "?pyplot.plot"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<a id=\"section2\"></a>\n",
    "### 2.2 Gradient Descent\n",
    "\n",
    "In this part, you will fit the linear regression parameters $\\theta$ to our dataset using gradient descent.\n",
    "\n",
    "#### 2.2.1 Update Equations\n",
    "\n",
    "The objective of linear regression is to minimize the cost function\n",
    "\n",
    "$$ J(\\theta) = \\frac{1}{2m} \\sum_{i=1}^m \\left( h_{\\theta}(x^{(i)}) - y^{(i)}\\right)^2$$\n",
    "\n",
    "where the hypothesis $h_\\theta(x)$ is given by the linear model\n",
    "$$ h_\\theta(x) = \\theta^Tx = \\theta_0 + \\theta_1 x_1$$\n",
    "\n",
    "Recall that the parameters of your model are the $\\theta_j$ values. These are\n",
    "the values you will adjust to minimize cost $J(\\theta)$. One way to do this is to\n",
    "use the batch gradient descent algorithm. In batch gradient descent, each\n",
    "iteration performs the update\n",
    "\n",
    "$$ \\theta_j = \\theta_j - \\alpha \\frac{1}{m} \\sum_{i=1}^m \\left( h_\\theta(x^{(i)}) - y^{(i)}\\right)x_j^{(i)} \\qquad \\text{simultaneously update } \\theta_j \\text{ for all } j$$\n",
    "\n",
    "With each step of gradient descent, your parameters $\\theta_j$ come closer to the optimal values that will achieve the lowest cost J($\\theta$).\n",
    "\n",
    "<div class=\"alert alert-block alert-warning\">\n",
    "**Implementation Note:** We store each example as a row in the the $X$ matrix in Python `numpy`. To take into account the intercept term ($\\theta_0$), we add an additional first column to $X$ and set it to all ones. This allows us to treat $\\theta_0$ as simply another 'feature'.\n",
    "</div>\n",
    "\n",
    "\n",
    "#### 2.2.2 Implementation\n",
    "\n",
    "We have already set up the data for linear regression. In the following cell, we add another dimension to our data to accommodate the $\\theta_0$ intercept term. Do NOT execute this cell more than once."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Add a column of ones to X. The numpy function stack joins arrays along a given axis. \n",
    "# The first axis (axis=0) refers to rows (training examples) \n",
    "# and second axis (axis=1) refers to columns (features).\n",
    "X = np.stack([np.ones(m), X], axis=1)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<a id=\"section2\"></a>\n",
    "#### 2.2.3 Computing the cost $J(\\theta)$\n",
    "\n",
    "As you perform gradient descent to learn minimize the cost function $J(\\theta)$, it is helpful to monitor the convergence by computing the cost. In this section, you will implement a function to calculate $J(\\theta)$ so you can check the convergence of your gradient descent implementation. \n",
    "\n",
    "Your next task is to complete the code for the function `computeCost` which computes $J(\\theta)$. As you are doing this, remember that the variables $X$ and $y$ are not scalar values. $X$ is a matrix whose rows represent the examples from the training set and $y$ is a vector whose each elemennt represent the value at a given row of $X$.\n",
    "<a id=\"computeCost\"></a>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [],
   "source": [
    "def computeCost(X, y, theta):\n",
    "    \"\"\"\n",
    "    Compute cost for linear regression. Computes the cost of using theta as the\n",
    "    parameter for linear regression to fit the data points in X and y.\n",
    "    \n",
    "    Parameters\n",
    "    ----------\n",
    "    X : array_like\n",
    "        The input dataset of shape (m x n+1), where m is the number of examples,\n",
    "        and n is the number of features. We assume a vector of one's already \n",
    "        appended to the features so we have n+1 columns.\n",
    "    \n",
    "    y : array_like\n",
    "        The values of the function at each data point. This is a vector of\n",
    "        shape (m, ).\n",
    "    \n",
    "    theta : array_like\n",
    "        The parameters for the regression function. This is a vector of \n",
    "        shape (n+1, ).\n",
    "    \n",
    "    Returns\n",
    "    -------\n",
    "    J : float\n",
    "        The value of the regression cost function.\n",
    "    \n",
    "    Instructions\n",
    "    ------------\n",
    "    Compute the cost of a particular choice of theta. \n",
    "    You should set J to the cost.\n",
    "    \"\"\"\n",
    "    \n",
    "    # initialize some useful values\n",
    "    m = y.size  # number of training examples\n",
    "    \n",
    "    # You need to return the following variables correctly\n",
    "    J = 0\n",
    "    \n",
    "    # ====================== YOUR CODE HERE =====================\n",
    "    k=np.dot(X,theta)-y\n",
    "    k=k**2\n",
    "    k=sum(k)\n",
    "    J=k/(2*m)\n",
    "\n",
    "    \n",
    "    # ===========================================================\n",
    "    return J"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Once you have completed the function, the next step will run `computeCost` two times using two different initializations of $\\theta$. You will see the cost printed to the screen."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "With theta = [0, 0] \n",
      "Cost computed = 32.07\n",
      "Expected cost value (approximately) 32.07\n",
      "\n",
      "With theta = [-1, 2]\n",
      "Cost computed = 54.24\n",
      "Expected cost value (approximately) 54.24\n"
     ]
    }
   ],
   "source": [
    "J = computeCost(X, y, theta=np.array([0.0, 0.0]))\n",
    "print('With theta = [0, 0] \\nCost computed = %.2f' % J)\n",
    "print('Expected cost value (approximately) 32.07\\n')\n",
    "\n",
    "# further testing of the cost function\n",
    "J = computeCost(X, y, theta=np.array([-1, 2]))\n",
    "print('With theta = [-1, 2]\\nCost computed = %.2f' % J)\n",
    "print('Expected cost value (approximately) 54.24')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "*You should now submit your solutions by executing the following cell.*"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "Submitting Solutions | Programming Exercise linear-regression\n",
      "\n",
      "Use token from last successful submission (k.kaushik25@gmail.com)? (Y/n): y\n",
      "                                  Part Name |     Score | Feedback\n",
      "                                  --------- |     ----- | --------\n",
      "                           Warm up exercise |  10 /  10 | Nice work!\n",
      "          Computing Cost (for one variable) |  40 /  40 | Nice work!\n",
      "        Gradient Descent (for one variable) |   0 /  50 | \n",
      "                      Feature Normalization |   0 /   0 | \n",
      "    Computing Cost (for multiple variables) |   0 /   0 | \n",
      "  Gradient Descent (for multiple variables) |   0 /   0 | \n",
      "                           Normal Equations |   0 /   0 | \n",
      "                                  --------------------------------\n",
      "                                            |  50 / 100 |  \n",
      "\n"
     ]
    }
   ],
   "source": [
    "grader[2] = computeCost\n",
    "grader.grade()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<a id=\"section3\"></a>\n",
    "#### 2.2.4 Gradient descent\n",
    "\n",
    "Next, you will complete a function which implements gradient descent.\n",
    "The loop structure has been written for you, and you only need to supply the updates to $\\theta$ within each iteration. \n",
    "\n",
    "As you program, make sure you understand what you are trying to optimize and what is being updated. Keep in mind that the cost $J(\\theta)$ is parameterized by the vector $\\theta$, not $X$ and $y$. That is, we minimize the value of $J(\\theta)$ by changing the values of the vector $\\theta$, not by changing $X$ or $y$. [Refer to the equations in this notebook](#section2) and to the video lectures if you are uncertain. A good way to verify that gradient descent is working correctly is to look at the value of $J(\\theta)$ and check that it is decreasing with each step. \n",
    "\n",
    "The starter code for the function `gradientDescent` calls `computeCost` on every iteration and saves the cost to a `python` list. Assuming you have implemented gradient descent and `computeCost` correctly, your value of $J(\\theta)$ should never increase, and should converge to a steady value by the end of the algorithm.\n",
    "\n",
    "<div class=\"alert alert-box alert-warning\">\n",
    "**Vectors and matrices in `numpy`** - Important implementation notes\n",
    "\n",
    "A vector in `numpy` is a one dimensional array, for example `np.array([1, 2, 3])` is a vector. A matrix in `numpy` is a two dimensional array, for example `np.array([[1, 2, 3], [4, 5, 6]])`. However, the following is still considered a matrix `np.array([[1, 2, 3]])` since it has two dimensions, even if it has a shape of 1x3 (which looks like a vector).\n",
    "\n",
    "Given the above, the function `np.dot` which we will use for all matrix/vector multiplication has the following properties:\n",
    "- It always performs inner products on vectors. If `x=np.array([1, 2, 3])`, then `np.dot(x, x)` is a scalar.\n",
    "- For matrix-vector multiplication, so if $X$ is a $m\\times n$ matrix and $y$ is a vector of length $m$, then the operation `np.dot(y, X)` considers $y$ as a $1 \\times m$ vector. On the other hand, if $y$ is a vector of length $n$, then the operation `np.dot(X, y)` considers $y$ as a $n \\times 1$ vector.\n",
    "- A vector can be promoted to a matrix using `y[None]` or `[y[np.newaxis]`. That is, if `y = np.array([1, 2, 3])` is a vector of size 3, then `y[None, :]` is a matrix of shape $1 \\times 3$. We can use `y[:, None]` to obtain a shape of $3 \\times 1$.\n",
    "<div>\n",
    "<a id=\"gradientDescent\"></a>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [],
   "source": [
    "def gradientDescent(X, y, theta, alpha, num_iters):\n",
    "    \"\"\"\n",
    "    Performs gradient descent to learn `theta`. Updates theta by taking `num_iters`\n",
    "    gradient steps with learning rate `alpha`.\n",
    "    \n",
    "    Parameters\n",
    "    ----------\n",
    "    X : array_like\n",
    "        The input dataset of shape (m x n+1).\n",
    "    \n",
    "    y : arra_like\n",
    "        Value at given features. A vector of shape (m, ).\n",
    "    \n",
    "    theta : array_like\n",
    "        Initial values for the linear regression parameters. \n",
    "        A vector of shape (n+1, ).\n",
    "    \n",
    "    alpha : float\n",
    "        The learning rate.\n",
    "    \n",
    "    num_iters : int\n",
    "        The number of iterations for gradient descent. \n",
    "    \n",
    "    Returns\n",
    "    -------\n",
    "    theta : array_like\n",
    "        The learned linear regression parameters. A vector of shape (n+1, ).\n",
    "    \n",
    "    J_history : list\n",
    "        A python list for the values of the cost function after each iteration.\n",
    "    \n",
    "    Instructions\n",
    "    ------------\n",
    "    Peform a single gradient step on the parameter vector theta.\n",
    "\n",
    "    While debugging, it can be useful to print out the values of \n",
    "    the cost function (computeCost) and gradient here.\n",
    "    \"\"\"\n",
    "    # Initialize some useful values\n",
    "    m = y.shape[0]  # number of training examples\n",
    "    \n",
    "    # make a copy of theta, to avoid changing the original array, since numpy arrays\n",
    "    # are passed by reference to functions\n",
    "    theta = theta.copy()\n",
    "    \n",
    "    J_history = [] # Use a python list to save cost in every iteration\n",
    "    \n",
    "    for i in range(num_iters):\n",
    "        # ==================== YOUR CODE HERE =================================\n",
    "        theta=theta-(alpha/m)*(X.T@(X@(theta)-y))\n",
    "        \n",
    "        \n",
    "\n",
    "        # =====================================================================\n",
    "        \n",
    "        # save the cost J in every iteration\n",
    "        J_history.append(computeCost(X, y, theta))\n",
    "    \n",
    "    return theta, J_history"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "After you are finished call the implemented `gradientDescent` function and print the computed $\\theta$. We initialize the $\\theta$ parameters to 0 and the learning rate $\\alpha$ to 0.01. Execute the following cell to check your code."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Theta found by gradient descent: -3.6303, 1.1664\n",
      "Expected theta values (approximately): [-3.6303, 1.1664]\n"
     ]
    }
   ],
   "source": [
    "# initialize fitting parameters\n",
    "theta = np.zeros(2)\n",
    "\n",
    "# some gradient descent settings\n",
    "iterations = 1500\n",
    "alpha = 0.01\n",
    "\n",
    "theta, J_history = gradientDescent(X ,y, theta, alpha, iterations)\n",
    "print('Theta found by gradient descent: {:.4f}, {:.4f}'.format(*theta))\n",
    "print('Expected theta values (approximately): [-3.6303, 1.1664]')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We will use your final parameters to plot the linear fit. The results should look like the following figure.\n",
    "\n",
    "![](Figures/regression_result.png)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAYIAAAEKCAYAAAAfGVI8AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDMuMC4yLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvOIA7rQAAIABJREFUeJztnXl4VNXZwH8nyQAZw4gSoFaEaKpWxICAFARbLVVBFq3GDQEXMEqAakQlqG2ttgJVpK2gKAFldcMNJan9xL2AFlBAASWjoCguoEIQDAN5vz/unTDJrElmy+T9Pc99MnPuufe8M3Nz3rO8ixERFEVRlKZLWqIFUBRFURKLKgJFUZQmjioCRVGUJo4qAkVRlCaOKgJFUZQmjioCRVGUJo4qAkVRlCaOKgJFUZQmjioCRVGUJk5GrG5sjDkGmAf8DKgCHhGRfxpj7gSuBb61q94mIqWh7pWdnS05OTmxElVRFCUlWb169Q4RaROuXswUAXAAGC8ia4wxLYHVxpj/s89NE5H7Ir1RTk4Oq1atiomQiqIoqYoxZmsk9WKmCERkO7Ddfl1hjNkIHB2r9hRFUZT6EZc9AmNMDnAq8I5dNNYYs84YM8cYc0Q8ZFAURVECE3NFYIzJAp4BbhSR3cBDQC7QFWvGMDXIdQXGmFXGmFXffvttoCqKoihKFIjlHgHGGAeWElgoIs8CiMjXPudnAS8FulZEHgEeAejRo4dfrGyPx8O2bdv46aefYiG6kiS0aNGC9u3b43A4Ei2KosQNt9vN9KlTWbRgATv27CE7K4uhw4Yxdvx4cnNzo95eLK2GDDAb2Cgi9/uUH2XvHwD8HvigPvfftm0bLVu2JCcnB6spJdUQEXbu3Mm2bds49thjEy2OosSFsrIyRuTnc63Hw3KPh47A1ooKZpeU0GvuXOYtXsyAAQOi2mYsZwR9gOHAemPM+3bZbcDlxpiugABbgOvqc/OffvpJlUCKY4yhdevW6NKg0lRwu92MyM9nyd699PYpzwXu8XgY7PEwJD+flevWRXVmEEurobeBQL10SJ+BuqBKIPXR31hpSkyfOpVrPZ4aSsCX3sAoj4cZ06Zx//TpUWu3SXgWu91uigoLaedykZ6WRjuXi6LCQtxud6JFUxRFqWbRggWM9HhC1hnl8bBo/vyotpvyiqCsrIxeeXlklpSwvKKCShGWV1SQWVJCr7w8ysrK6nXfnTt30rVrV7p27crPfvYzjj766Or3+/fvj+geV199NR999FHIOjNmzGDhwoX1kjEUr7zyChdccEHIOmvWrOHf//531NtWFCUwO/bsoWOYOh3setEkplZDiSaW622tW7fm/fetrY8777yTrKwsbr755hp1RAQRIS0tsL599NFHw7YzZsyYOskVTdasWcMHH3xA//79EyaDojQlsrOy2FpRQaje6DO7XjRJ6RlBXdbbokV5eTmdO3fm+uuvp1u3bmzfvp2CggJ69OjBySefzF133VVdt2/fvrz//vscOHCAVq1aUVxcTJcuXejduzfffPMNAHfccQf/+Mc/qusXFxfTs2dPTjzxRJYvXw7Ajz/+yEUXXUSXLl24/PLL6dGjR7WS8mXp0qWceOKJ9O3blxdeeKG6fOXKlfTu3ZtTTz2VPn36sHnzZvbt28ddd93FwoUL6dq1K4sXLw5YT1GU6DF02DBmhzGVLnE4GDp8eHQb9o5ak/no3r271GbDhg1+ZbVp27KllINIiKMcpJ3LFfZeofjzn/8s9957r4iIbN68WYwx8u6771af37lzp4iIeDwe6du3r3z44YciItKnTx957733xOPxCCClpaUiIlJUVCSTJk0SEZHbb79dpk2bVl3/1ltvFRGRF154Qc4991wREZk0aZIUFhaKiMj7778vaWlp8t5779WQ8ccff5Sjjz5aysvLpaqqSi688EI5//zzRUTkhx9+kAMHDoiISFlZmVxyySUiIjJr1iy54YYbqu8RrF6sieS3VpRUoLy8XLKdTlkepL9aDpLtdEp5eXlE9wNWSQR9bEovDSVqvS03N5fTTjut+v3jjz/O7NmzOXDgAF9++SUbNmygU6dONa7JzMystg3u3r07b731VsB7X3jhhdV1tmzZAsDbb7/NhAkTAOjSpQsnn3yy33UbNmzghBNOqF4Cu+KKK5g3bx4AP/zwAyNGjAi7eR5pPUVR6kdubi7zFi9mSH4+ozweRnk8dMBaDipxOChxOJi3eHHUncpSemkoOyuLcKH3YrHedthhh1W/3rx5M//85z959dVXWbduHf379w/oDd2sWbPq1+np6Rw4cCDgvZs3b+5Xx1L84Qlminn77bdz7rnn8sEHH/D8888H9daOtJ6iKPVnwIABrFy3jsqCAvq4XGSmpdHH5aKyoICV69ZF3ZkMUlwRJGy9zYfdu3fTsmVLXC4X27dv5+WXX456G3379uWpp54CYP369WzYsMGvTqdOnfj444/59NNPEREef/zx6nO7du3i6KOtwLCPPfZYdXnLli2pqKgIW09R4kVTMQXPzc3l/unT+WrXLg4cPMhXu3Zx//TpMQkvASmuCMaOH88sh4MVQc6vwFIEY4qKYiZDt27d6NSpE507d+baa6+lT58+UW9j3LhxfPHFF+Tl5TF16lQ6d+7M4YcfXqOO0+lk5syZDBgwgDPOOIPjjjuu+tyECRO45ZZb/GT77W9/y9q1azn11FNZvHhx0HqKEg9iZQqukNqbxSIipaWlku10SrHDIeUg++0N4mKHQ7KdzuoN2saMx+ORffv2iYjIxx9/LDk5OeLxeBIsVfTQzWIl2puoTQUi3CxO6RkBJGa9Ld7s2bOHPn360KVLFy666CIefvhhMjJS2g5AaWIkwhS8KWEkwo3GRNKjRw+pnapy48aNnHTSSQmSSIkn+lsr7VwulodxtHIDfVwuvtq1K15iJT3GmNUi0iNcvZSfESiK0vhJlCl4U0EVgaIoSU+iTMGbCqoIFEVJeuJhCp5spqkzXitn+Ox3+LEysE9RNFFFoChK0hNrU/BkMU0VESaXbSKneCn3vvwRb23eoYog2ckKMA2dOXNmdeiGpsySJUuYPHlyosVQUoTq0AtOJxMdDtyAB2uDeKLDwRCns96hF3yjFN/j8ZCLFZbZG6V4yd69jMjPj+nMoKpKuO259Rw7sZSZb1jtdGzt5L0/nk1bV4uYtetFbQyjzPXXXx/T+1fb/QYJbX3w4EHS09Prff+GXu9lyJAhDBkypMH3URQvXlPwGdOm0Wf+/ENJ3YcPZ2VRUb29bhOVFQzgwMEqbnpqLUvWfllddsrRh7Po2l/RskXopbBoojOCKHPnnXdy3333AXDmmWcyYcIEevbsyQknnFAdSO7gwYPccsstnHbaaeTl5fHwww8Dlj9Av3796NatG6ecckp1qOgtW7Zw0kknUVhYSLdu3fj8889rtJmTk8Ndd91F3759efrpp3G73fTv35/u3btzxhlnsGnTJsAa+fTq1YvTTjuNP/3pT9Uzmtdff52zzjqLoUOHcsoppwCwYMECevbsSdeuXbnuuus4ePAgBw8e5KqrrqJz586ccsopTLNttv/1r3/RqVMn8vLyuOyyywArBMXYsWMB2Lp1K/369SMvL49+/frx2WefAXDVVVfxhz/8gdNPP53jjjuOxYsXx+ZHUVKGWIReSERWsMoDB7nq0Xf5xe1l1Urg9NzWbLq7Py+O6xtXJQApMiP4y4sfsuHL3VG9Z6efu/jzYP8onnXlwIEDvPvuu5SWlvKXv/yFV155hdmzZ3P44Yfzv//9j8rKSvr06cM555zDMcccw3PPPYfL5WLHjh306tWrelT90Ucf8eijj/Lggw8GbKdFixa8/fbbAPTr14+ZM2dy/PHH884771BYWMirr77KDTfcwA033MDll1/OzJkza1z/7rvv8sEHH3DssceyceNGnnzySf773//icDgoLCxk4cKFnHzyyXzxxRd88MEHgBWNFGDy5Ml8+umnNG/evLrMl7FjxzJixAiuvPJK5syZwx/+8Aeef/55ALZv387bb7/Npk2bGDJkCPn5+Q3+zhWlLtQ2TXUD04FFwA4gG7gM+NYn7lZ92bv/AMNK3mHNZ4f+T849uR0PXN6NZhmJG5enhCJIZgKFjf7Pf/7DunXrqkfAu3btYvPmzbRv357bbruNN998k7S0NL744gu+/vprADp27EivXr2CtnPppZcC1qxi+fLlXHzxxdXnKisrAVixYkV1Bzx06NAaGdV69uzJscceC8CyZctYvXp1dSjtffv20bZtWwYPHswnn3zCuHHjGDhwIOeccw4AeXl5XHHFFVxwwQUB01+uWLGCZ599FoDhw4dz6623Vp+74IILSEtLo1OnTtWfVVHiiW9WsDJgBHAtsBzoCGwFSoAWIpSVldUrGsGufR4uemg55d8c8nPI796eKRflkZ4WOCpwPEkJRRCNkXusCBY2+oEHHuDcc8+tUfexxx7j22+/ZfXq1TgcDnJycqpDPfuGtg6E93xVVRWtWrUKmKEskuu98l155ZVMmjTJr97atWt5+eWXmTFjBk899RRz5sxh6dKlvPnmmyxZsoS7776bDz/8MGRbvuGwvd+Pt11FiTdDhw1jdkkJIz0eRgBLwC+17SRgCNQ5te2OPZUM+tfbfLX7UMj2q/vk8KdBnYKGhU8EukeQAM4991weeughPPa65Mcff8yPP/7Irl27aNu2LQ6Hg9dee42tW8O50Pjjcrk49thjefrppwGrc127di0AvXr14plnngHgiSeeCHqP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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# plot the linear fit\n",
    "plotData(X[:, 1], y)\n",
    "pyplot.plot(X[:, 1], np.dot(X, theta), '-')\n",
    "pyplot.legend(['Training data', 'Linear regression']);"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Your final values for $\\theta$ will also be used to make predictions on profits in areas of 35,000 and 70,000 people.\n",
    "\n",
    "<div class=\"alert alert-block alert-success\">\n",
    "Note the way that the following lines use matrix multiplication, rather than explicit summation or looping, to calculate the predictions. This is an example of code vectorization in `numpy`.\n",
    "</div>\n",
    "\n",
    "<div class=\"alert alert-block alert-success\">\n",
    "Note that the first argument to the `numpy` function `dot` is a python list. `numpy` can internally converts **valid** python lists to numpy arrays when explicitly provided as arguments to `numpy` functions.\n",
    "</div>\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "For population = 35,000, we predict a profit of 4519.77\n",
      "\n",
      "For population = 70,000, we predict a profit of 45342.45\n",
      "\n"
     ]
    }
   ],
   "source": [
    "# Predict values for population sizes of 35,000 and 70,000\n",
    "predict1 = np.dot([1, 3.5], theta)\n",
    "print('For population = 35,000, we predict a profit of {:.2f}\\n'.format(predict1*10000))\n",
    "\n",
    "predict2 = np.dot([1, 7], theta)\n",
    "print('For population = 70,000, we predict a profit of {:.2f}\\n'.format(predict2*10000))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "*You should now submit your solutions by executing the next cell.*"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "Submitting Solutions | Programming Exercise linear-regression\n",
      "\n",
      "Use token from last successful submission (k.kaushik25@gmail.com)? (Y/n): y\n",
      "                                  Part Name |     Score | Feedback\n",
      "                                  --------- |     ----- | --------\n",
      "                           Warm up exercise |  10 /  10 | Nice work!\n",
      "          Computing Cost (for one variable) |  40 /  40 | Nice work!\n",
      "        Gradient Descent (for one variable) |  50 /  50 | Nice work!\n",
      "                      Feature Normalization |   0 /   0 | \n",
      "    Computing Cost (for multiple variables) |   0 /   0 | \n",
      "  Gradient Descent (for multiple variables) |   0 /   0 | \n",
      "                           Normal Equations |   0 /   0 | \n",
      "                                  --------------------------------\n",
      "                                            | 100 / 100 |  \n",
      "\n"
     ]
    }
   ],
   "source": [
    "grader[3] = gradientDescent\n",
    "grader.grade()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 2.4 Visualizing $J(\\theta)$\n",
    "\n",
    "To understand the cost function $J(\\theta)$ better, you will now plot the cost over a 2-dimensional grid of $\\theta_0$ and $\\theta_1$ values. You will not need to code anything new for this part, but you should understand how the code you have written already is creating these images.\n",
    "\n",
    "In the next cell, the code is set up to calculate $J(\\theta)$ over a grid of values using the `computeCost` function that you wrote. After executing the following cell, you will have a 2-D array of $J(\\theta)$ values. Then, those values are used to produce surface and contour plots of $J(\\theta)$ using the matplotlib `plot_surface` and `contourf` functions. The plots should look something like the following:\n",
    "\n",
    "![](Figures/cost_function.png)\n",
    "\n",
    "The purpose of these graphs is to show you how $J(\\theta)$ varies with changes in $\\theta_0$ and $\\theta_1$. The cost function $J(\\theta)$ is bowl-shaped and has a global minimum. (This is easier to see in the contour plot than in the 3D surface plot). This minimum is the optimal point for $\\theta_0$ and $\\theta_1$, and each step of gradient descent moves closer to this point."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# grid over which we will calculate J\n",
    "theta0_vals = np.linspace(-10, 10, 100)\n",
    "theta1_vals = np.linspace(-1, 4, 100)\n",
    "\n",
    "# initialize J_vals to a matrix of 0's\n",
    "J_vals = np.zeros((theta0_vals.shape[0], theta1_vals.shape[0]))\n",
    "\n",
    "# Fill out J_vals\n",
    "for i, theta0 in enumerate(theta0_vals):\n",
    "    for j, theta1 in enumerate(theta1_vals):\n",
    "        J_vals[i, j] = computeCost(X, y, [theta0, theta1])\n",
    "        \n",
    "# Because of the way meshgrids work in the surf command, we need to\n",
    "# transpose J_vals before calling surf, or else the axes will be flipped\n",
    "J_vals = J_vals.T\n",
    "\n",
    "# surface plot\n",
    "fig = pyplot.figure(figsize=(12, 5))\n",
    "ax = fig.add_subplot(121, projection='3d')\n",
    "ax.plot_surface(theta0_vals, theta1_vals, J_vals, cmap='viridis')\n",
    "pyplot.xlabel('theta0')\n",
    "pyplot.ylabel('theta1')\n",
    "pyplot.title('Surface')\n",
    "\n",
    "# contour plot\n",
    "# Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100\n",
    "ax = pyplot.subplot(122)\n",
    "pyplot.contour(theta0_vals, theta1_vals, J_vals, linewidths=2, cmap='viridis', levels=np.logspace(-2, 3, 20))\n",
    "pyplot.xlabel('theta0')\n",
    "pyplot.ylabel('theta1')\n",
    "pyplot.plot(theta[0], theta[1], 'ro', ms=10, lw=2)\n",
    "pyplot.title('Contour, showing minimum')\n",
    "pass"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Optional Exercises\n",
    "\n",
    "If you have successfully completed the material above, congratulations! You now understand linear regression and should able to start using it on your own datasets.\n",
    "\n",
    "For the rest of this programming exercise, we have included the following optional exercises. These exercises will help you gain a deeper understanding of the material, and if you are able to do so, we encourage you to complete them as well. You can still submit your solutions to these exercises to check if your answers are correct.\n",
    "\n",
    "## 3 Linear regression with multiple variables\n",
    "\n",
    "In this part, you will implement linear regression with multiple variables to predict the prices of houses. Suppose you are selling your house and you want to know what a good market price would be. One way to do this is to first collect information on recent houses sold and make a model of housing prices.\n",
    "\n",
    "The file `Data/ex1data2.txt` contains a training set of housing prices in Portland, Oregon. The first column is the size of the house (in square feet), the second column is the number of bedrooms, and the third column is the price\n",
    "of the house. \n",
    "\n",
    "<a id=\"section4\"></a>\n",
    "### 3.1 Feature Normalization\n",
    "\n",
    "We start by loading and displaying some values from this dataset. By looking at the values, note that house sizes are about 1000 times the number of bedrooms. When features differ by orders of magnitude, first performing feature scaling can make gradient descent converge much more quickly."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load data\n",
    "data = np.loadtxt(os.path.join('Data', 'ex1data2.txt'), delimiter=',')\n",
    "X = data[:, :2]\n",
    "y = data[:, 2]\n",
    "m = y.size\n",
    "\n",
    "# print out some data points\n",
    "print('{:>8s}{:>8s}{:>10s}'.format('X[:,0]', 'X[:, 1]', 'y'))\n",
    "print('-'*26)\n",
    "for i in range(10):\n",
    "    print('{:8.0f}{:8.0f}{:10.0f}'.format(X[i, 0], X[i, 1], y[i]))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Your task here is to complete the code in `featureNormalize` function:\n",
    "- Subtract the mean value of each feature from the dataset.\n",
    "- After subtracting the mean, additionally scale (divide) the feature values by their respective â€œstandard deviations.â€\n",
    "\n",
    "The standard deviation is a way of measuring how much variation there is in the range of values of a particular feature (most data points will lie within Â±2 standard deviations of the mean); this is an alternative to taking the range of values (max-min). In `numpy`, you can use the `std` function to compute the standard deviation. \n",
    "\n",
    "For example, the quantity `X[:, 0]` contains all the values of $x_1$ (house sizes) in the training set, so `np.std(X[:, 0])` computes the standard deviation of the house sizes.\n",
    "At the time that the function `featureNormalize` is called, the extra column of 1â€™s corresponding to $x_0 = 1$ has not yet been added to $X$. \n",
    "\n",
    "You will do this for all the features and your code should work with datasets of all sizes (any number of features / examples). Note that each column of the matrix $X$ corresponds to one feature.\n",
    "\n",
    "<div class=\"alert alert-block alert-warning\">\n",
    "**Implementation Note:** When normalizing the features, it is important\n",
    "to store the values used for normalization - the mean value and the standard deviation used for the computations. After learning the parameters\n",
    "from the model, we often want to predict the prices of houses we have not\n",
    "seen before. Given a new x value (living room area and number of bedrooms), we must first normalize x using the mean and standard deviation that we had previously computed from the training set.\n",
    "</div>\n",
    "<a id=\"featureNormalize\"></a>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [],
   "source": [
    "def  featureNormalize(X):\n",
    "    \"\"\"\n",
    "    Normalizes the features in X. returns a normalized version of X where\n",
    "    the mean value of each feature is 0 and the standard deviation\n",
    "    is 1. This is often a good preprocessing step to do when working with\n",
    "    learning algorithms.\n",
    "    \n",
    "    Parameters\n",
    "    ----------\n",
    "    X : array_like\n",
    "        The dataset of shape (m x n).\n",
    "    \n",
    "    Returns\n",
    "    -------\n",
    "    X_norm : array_like\n",
    "        The normalized dataset of shape (m x n).\n",
    "    \n",
    "    Instructions\n",
    "    ------------\n",
    "    First, for each feature dimension, compute the mean of the feature\n",
    "    and subtract it from the dataset, storing the mean value in mu. \n",
    "    Next, compute the  standard deviation of each feature and divide\n",
    "    each feature by it's standard deviation, storing the standard deviation \n",
    "    in sigma. \n",
    "    \n",
    "    Note that X is a matrix where each column is a feature and each row is\n",
    "    an example. You needto perform the normalization separately for each feature. \n",
    "    \n",
    "    Hint\n",
    "    ----\n",
    "    You might find the 'np.mean' and 'np.std' functions useful.\n",
    "    \"\"\"\n",
    "    # You need to set these values correctly\n",
    "    X_norm = X.copy()\n",
    "    mu = np.zeros(X.shape[1])\n",
    "    sigma = np.zeros(X.shape[1])\n",
    "\n",
    "    # =========================== YOUR CODE HERE =====================\n",
    "    for i in range(len(X[0,:])):\n",
    "        mu[i]=np.mean(X[:,i])\n",
    "        sigma[i]=np.std(X[:,i])\n",
    "        X_norm[:,i]=(X_norm[:,i]-mu[i])/sigma[i]\n",
    "\n",
    "    \n",
    "    # ================================================================\n",
    "    return X_norm, mu, sigma"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Execute the next cell to run the implemented `featureNormalize` function."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Computed mean: [1.     8.1598]\n",
      "Computed standard deviation: [0.       3.849884]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "c:\\users\\administrator\\appdata\\local\\programs\\python\\python37-32\\lib\\site-packages\\ipykernel_launcher.py:42: RuntimeWarning: invalid value encountered in true_divide\n"
     ]
    }
   ],
   "source": [
    "# call featureNormalize on the loaded data\n",
    "X_norm, mu, sigma = featureNormalize(X)\n",
    "\n",
    "print('Computed mean:', mu)\n",
    "print('Computed standard deviation:', sigma)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "*You should not submit your solutions.*"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "Submitting Solutions | Programming Exercise linear-regression\n",
      "\n",
      "Use token from last successful submission (k.kaushik25@gmail.com)? (Y/n): y\n",
      "                                  Part Name |     Score | Feedback\n",
      "                                  --------- |     ----- | --------\n",
      "                           Warm up exercise |  10 /  10 | Nice work!\n",
      "          Computing Cost (for one variable) |  40 /  40 | Nice work!\n",
      "        Gradient Descent (for one variable) |  50 /  50 | Nice work!\n",
      "                      Feature Normalization |   0 /   0 | Nice work!\n",
      "    Computing Cost (for multiple variables) |   0 /   0 | \n",
      "  Gradient Descent (for multiple variables) |   0 /   0 | \n",
      "                           Normal Equations |   0 /   0 | \n",
      "                                  --------------------------------\n",
      "                                            | 100 / 100 |  \n",
      "\n"
     ]
    }
   ],
   "source": [
    "grader[4] = featureNormalize\n",
    "grader.grade()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "After the `featureNormalize` function is tested, we now add the intercept term to `X_norm`:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Add intercept term to X\n",
    "X = np.concatenate([np.ones((m, 1)), X_norm], axis=1)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<a id=\"section5\"></a>\n",
    "### 3.2 Gradient Descent\n",
    "\n",
    "Previously, you implemented gradient descent on a univariate regression problem. The only difference now is that there is one more feature in the matrix $X$. The hypothesis function and the batch gradient descent update\n",
    "rule remain unchanged. \n",
    "\n",
    "You should complete the code for the functions `computeCostMulti` and `gradientDescentMulti` to implement the cost function and gradient descent for linear regression with multiple variables. If your code in the previous part (single variable) already supports multiple variables, you can use it here too.\n",
    "Make sure your code supports any number of features and is well-vectorized.\n",
    "You can use the `shape` property of `numpy` arrays to find out how many features are present in the dataset.\n",
    "\n",
    "<div class=\"alert alert-block alert-warning\">\n",
    "**Implementation Note:** In the multivariate case, the cost function can\n",
    "also be written in the following vectorized form:\n",
    "\n",
    "$$ J(\\theta) = \\frac{1}{2m}(X\\theta - \\vec{y})^T(X\\theta - \\vec{y}) $$\n",
    "\n",
    "where \n",
    "\n",
    "$$ X = \\begin{pmatrix}\n",
    "          - (x^{(1)})^T - \\\\\n",
    "          - (x^{(2)})^T - \\\\\n",
    "          \\vdots \\\\\n",
    "          - (x^{(m)})^T - \\\\ \\\\\n",
    "        \\end{pmatrix} \\qquad \\mathbf{y} = \\begin{bmatrix} y^{(1)} \\\\ y^{(2)} \\\\ \\vdots \\\\ y^{(m)} \\\\\\end{bmatrix}$$\n",
    "\n",
    "the vectorized version is efficient when you are working with numerical computing tools like `numpy`. If you are an expert with matrix operations, you can prove to yourself that the two forms are equivalent.\n",
    "</div>\n",
    "\n",
    "<a id=\"computeCostMulti\"></a>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [],
   "source": [
    "def computeCostMulti(X, y, theta):\n",
    "    \"\"\"\n",
    "    Compute cost for linear regression with multiple variables.\n",
    "    Computes the cost of using theta as the parameter for linear regression to fit the data points in X and y.\n",
    "    \n",
    "    Parameters\n",
    "    ----------\n",
    "    X : array_like\n",
    "        The dataset of shape (m x n+1).\n",
    "    \n",
    "    y : array_like\n",
    "        A vector of shape (m, ) for the values at a given data point.\n",
    "    \n",
    "    theta : array_like\n",
    "        The linear regression parameters. A vector of shape (n+1, )\n",
    "    \n",
    "    Returns\n",
    "    -------\n",
    "    J : float\n",
    "        The value of the cost function. \n",
    "    \n",
    "    Instructions\n",
    "    ------------\n",
    "    Compute the cost of a particular choice of theta. You should set J to the cost.\n",
    "    \"\"\"\n",
    "    # Initialize some useful values\n",
    "    m = y.shape[0] # number of training examples\n",
    "    \n",
    "    # You need to return the following variable correctly\n",
    "    J = 0\n",
    "    \n",
    "    # ======================= YOUR CODE HERE ===========================\n",
    "    k=np.dot(X,theta)-y\n",
    "    k=k**2\n",
    "    k=sum(k)\n",
    "    J=k/(2*m)\n",
    "    \n",
    "\n",
    "    \n",
    "    # ==================================================================\n",
    "    return J\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "*You should now submit your solutions.*"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "Submitting Solutions | Programming Exercise linear-regression\n",
      "\n",
      "Use token from last successful submission (k.kaushik25@gmail.com)? (Y/n): y\n",
      "                                  Part Name |     Score | Feedback\n",
      "                                  --------- |     ----- | --------\n",
      "                           Warm up exercise |  10 /  10 | Nice work!\n",
      "          Computing Cost (for one variable) |  40 /  40 | Nice work!\n",
      "        Gradient Descent (for one variable) |  50 /  50 | Nice work!\n",
      "                      Feature Normalization |   0 /   0 | Nice work!\n",
      "    Computing Cost (for multiple variables) |   0 /   0 | Nice work!\n",
      "  Gradient Descent (for multiple variables) |   0 /   0 | \n",
      "                           Normal Equations |   0 /   0 | \n",
      "                                  --------------------------------\n",
      "                                            | 100 / 100 |  \n",
      "\n"
     ]
    }
   ],
   "source": [
    "grader[5] = computeCostMulti\n",
    "grader.grade()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<a id=\"gradientDescentMulti\"></a>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [],
   "source": [
    "def gradientDescentMulti(X, y, theta, alpha, num_iters):\n",
    "    \"\"\"\n",
    "    Performs gradient descent to learn theta.\n",
    "    Updates theta by taking num_iters gradient steps with learning rate alpha.\n",
    "        \n",
    "    Parameters\n",
    "    ----------\n",
    "    X : array_like\n",
    "        The dataset of shape (m x n+1).\n",
    "    \n",
    "    y : array_like\n",
    "        A vector of shape (m, ) for the values at a given data point.\n",
    "    \n",
    "    theta : array_like\n",
    "        The linear regression parameters. A vector of shape (n+1, )\n",
    "    \n",
    "    alpha : float\n",
    "        The learning rate for gradient descent. \n",
    "    \n",
    "    num_iters : int\n",
    "        The number of iterations to run gradient descent. \n",
    "    \n",
    "    Returns\n",
    "    -------\n",
    "    theta : array_like\n",
    "        The learned linear regression parameters. A vector of shape (n+1, ).\n",
    "    \n",
    "    J_history : list\n",
    "        A python list for the values of the cost function after each iteration.\n",
    "    \n",
    "    Instructions\n",
    "    ------------\n",
    "    Peform a single gradient step on the parameter vector theta.\n",
    "\n",
    "    While debugging, it can be useful to print out the values of \n",
    "    the cost function (computeCost) and gradient here.\n",
    "    \"\"\"\n",
    "    # Initialize some useful values\n",
    "    m = y.shape[0] # number of training examples\n",
    "    \n",
    "    # make a copy of theta, which will be updated by gradient descent\n",
    "    theta = theta.copy()\n",
    "    \n",
    "    J_history = []\n",
    "    \n",
    "    for i in range(num_iters):\n",
    "        # ======================= YOUR CODE HERE ==========================\n",
    "        theta=theta-(alpha/m)*(X.T@(X@(theta)-y))\n",
    "\n",
    "        \n",
    "        # =================================================================\n",
    "        \n",
    "        # save the cost J in every iteration\n",
    "        J_history.append(computeCostMulti(X, y, theta))\n",
    "    \n",
    "    return theta, J_history"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "*You should now submit your solutions.*"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "Submitting Solutions | Programming Exercise linear-regression\n",
      "\n",
      "Use token from last successful submission (k.kaushik25@gmail.com)? (Y/n): y\n",
      "                                  Part Name |     Score | Feedback\n",
      "                                  --------- |     ----- | --------\n",
      "                           Warm up exercise |  10 /  10 | Nice work!\n",
      "          Computing Cost (for one variable) |  40 /  40 | Nice work!\n",
      "        Gradient Descent (for one variable) |  50 /  50 | Nice work!\n",
      "                      Feature Normalization |   0 /   0 | Nice work!\n",
      "    Computing Cost (for multiple variables) |   0 /   0 | Nice work!\n",
      "  Gradient Descent (for multiple variables) |   0 /   0 | Nice work!\n",
      "                           Normal Equations |   0 /   0 | \n",
      "                                  --------------------------------\n",
      "                                            | 100 / 100 |  \n",
      "\n"
     ]
    }
   ],
   "source": [
    "grader[6] = gradientDescentMulti\n",
    "grader.grade()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### 3.2.1 Optional (ungraded) exercise: Selecting learning rates\n",
    "\n",
    "In this part of the exercise, you will get to try out different learning rates for the dataset and find a learning rate that converges quickly. You can change the learning rate by modifying the following code and changing the part of the code that sets the learning rate.\n",
    "\n",
    "Use your implementation of `gradientDescentMulti` function and run gradient descent for about 50 iterations at the chosen learning rate. The function should also return the history of $J(\\theta)$ values in a vector $J$.\n",
    "\n",
    "After the last iteration, plot the J values against the number of the iterations.\n",
    "\n",
    "If you picked a learning rate within a good range, your plot look similar as the following Figure. \n",
    "\n",
    "![](Figures/learning_rate.png)\n",
    "\n",
    "If your graph looks very different, especially if your value of $J(\\theta)$ increases or even blows up, adjust your learning rate and try again. We recommend trying values of the learning rate $\\alpha$ on a log-scale, at multiplicative steps of about 3 times the previous value (i.e., 0.3, 0.1, 0.03, 0.01 and so on). You may also want to adjust the number of iterations you are running if that will help you see the overall trend in the curve.\n",
    "\n",
    "<div class=\"alert alert-block alert-warning\">\n",
    "**Implementation Note:** If your learning rate is too large, $J(\\theta)$ can diverge and â€˜blow upâ€™, resulting in values which are too large for computer calculations. In these situations, `numpy` will tend to return\n",
    "NaNs. NaN stands for â€˜not a numberâ€™ and is often caused by undefined operations that involve âˆ’âˆž and +âˆž.\n",
    "</div>\n",
    "\n",
    "<div class=\"alert alert-block alert-warning\">\n",
    "**MATPLOTLIB tip:** To compare how different learning learning rates affect convergence, it is helpful to plot $J$ for several learning rates on the same figure. This can be done by making `alpha` a python list, and looping across the values within this list, and calling the plot function in every iteration of the loop. It is also useful to have a legend to distinguish the different lines within the plot. Search online for `pyplot.legend` for help on showing legends in `matplotlib`.\n",
    "</div>\n",
    "\n",
    "Notice the changes in the convergence curves as the learning rate changes. With a small learning rate, you should find that gradient descent takes a very long time to converge to the optimal value. Conversely, with a large learning rate, gradient descent might not converge or might even diverge!\n",
    "Using the best learning rate that you found, run the script\n",
    "to run gradient descent until convergence to find the final values of $\\theta$. Next,\n",
    "use this value of $\\theta$ to predict the price of a house with 1650 square feet and\n",
    "3 bedrooms. You will use value later to check your implementation of the normal equations. Donâ€™t forget to normalize your features when you make this prediction!"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {},
   "outputs": [
    {
     "ename": "ValueError",
     "evalue": "matmul: Input operand 1 has a mismatch in its core dimension 0, with gufunc signature (n?,k),(k,m?)->(n?,m?) (size 3 is different from 2)",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mValueError\u001b[0m                                Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-34-27ce263f4731>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[0;32m     25\u001b[0m \u001b[1;31m# init theta and run gradient descent\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m     26\u001b[0m \u001b[0mtheta\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mnp\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mzeros\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;36m3\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m---> 27\u001b[1;33m \u001b[0mtheta\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mJ_history\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mgradientDescentMulti\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mX\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0my\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mtheta\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0malpha\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mnum_iters\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m     28\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m     29\u001b[0m \u001b[1;31m# Plot the convergence graph\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m<ipython-input-31-99f9042ffccf>\u001b[0m in \u001b[0;36mgradientDescentMulti\u001b[1;34m(X, y, theta, alpha, num_iters)\u001b[0m\n\u001b[0;32m     46\u001b[0m     \u001b[1;32mfor\u001b[0m \u001b[0mi\u001b[0m \u001b[1;32min\u001b[0m \u001b[0mrange\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mnum_iters\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m     47\u001b[0m         \u001b[1;31m# ======================= YOUR CODE HERE ==========================\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m---> 48\u001b[1;33m         \u001b[0mtheta\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mtheta\u001b[0m\u001b[1;33m-\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0malpha\u001b[0m\u001b[1;33m/\u001b[0m\u001b[0mm\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m*\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mX\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mT\u001b[0m\u001b[1;33m@\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mX\u001b[0m\u001b[1;33m@\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mtheta\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m-\u001b[0m\u001b[0my\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m     49\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m     50\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;31mValueError\u001b[0m: matmul: Input operand 1 has a mismatch in its core dimension 0, with gufunc signature (n?,k),(k,m?)->(n?,m?) (size 3 is different from 2)"
     ]
    }
   ],
   "source": [
    "\"\"\"\n",
    "Instructions\n",
    "------------\n",
    "We have provided you with the following starter code that runs\n",
    "gradient descent with a particular learning rate (alpha). \n",
    "\n",
    "Your task is to first make sure that your functions - `computeCost`\n",
    "and `gradientDescent` already work with  this starter code and\n",
    "support multiple variables.\n",
    "\n",
    "After that, try running gradient descent with different values of\n",
    "alpha and see which one gives you the best result.\n",
    "\n",
    "Finally, you should complete the code at the end to predict the price\n",
    "of a 1650 sq-ft, 3 br house.\n",
    "\n",
    "Hint\n",
    "----\n",
    "At prediction, make sure you do the same feature normalization.\n",
    "\"\"\"\n",
    "# Choose some alpha value - change this\n",
    "alpha = 0.1\n",
    "num_iters = 400\n",
    "\n",
    "# init theta and run gradient descent\n",
    "theta = np.zeros(3)\n",
    "theta, J_history = gradientDescentMulti(X, y, theta, alpha, num_iters)\n",
    "\n",
    "# Plot the convergence graph\n",
    "pyplot.plot(np.arange(len(J_history)), J_history, lw=2)\n",
    "pyplot.xlabel('Number of iterations')\n",
    "pyplot.ylabel('Cost J')\n",
    "\n",
    "# Display the gradient descent's result\n",
    "print('theta computed from gradient descent: {:s}'.format(str(theta)))\n",
    "\n",
    "# Estimate the price of a 1650 sq-ft, 3 br house\n",
    "# ======================= YOUR CODE HERE ===========================\n",
    "# Recall that the first column of X is all-ones. \n",
    "# Thus, it does not need to be normalized.\n",
    "\n",
    "price = 0   # You should change this\n",
    "\n",
    "# ===================================================================\n",
    "\n",
    "print('Predicted price of a 1650 sq-ft, 3 br house (using gradient descent): ${:.0f}'.format(price))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "*You do not need to submit any solutions for this optional (ungraded) part.*"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<a id=\"section7\"></a>\n",
    "### 3.3 Normal Equations\n",
    "\n",
    "In the lecture videos, you learned that the closed-form solution to linear regression is\n",
    "\n",
    "$$ \\theta = \\left( X^T X\\right)^{-1} X^T\\vec{y}$$\n",
    "\n",
    "Using this formula does not require any feature scaling, and you will get an exact solution in one calculation: there is no â€œloop until convergenceâ€ like in gradient descent. \n",
    "\n",
    "First, we will reload the data to ensure that the variables have not been modified. Remember that while you do not need to scale your features, we still need to add a column of 1â€™s to the $X$ matrix to have an intercept term ($\\theta_0$). The code in the next cell will add the column of 1â€™s to X for you."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load data\n",
    "data = np.loadtxt(os.path.join('Data', 'ex1data2.txt'), delimiter=',')\n",
    "X = data[:, :2]\n",
    "y = data[:, 2]\n",
    "m = y.size\n",
    "X = np.concatenate([np.ones((m, 1)), X], axis=1)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Complete the code for the function `normalEqn` below to use the formula above to calculate $\\theta$. \n",
    "\n",
    "<a id=\"normalEqn\"></a>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {},
   "outputs": [],
   "source": [
    "def normalEqn(X, y):\n",
    "    \"\"\"\n",
    "    Computes the closed-form solution to linear regression using the normal equations.\n",
    "    \n",
    "    Parameters\n",
    "    ----------\n",
    "    X : array_like\n",
    "        The dataset of shape (m x n+1).\n",
    "    \n",
    "    y : array_like\n",
    "        The value at each data point. A vector of shape (m, ).\n",
    "    \n",
    "    Returns\n",
    "    -------\n",
    "    theta : array_like\n",
    "        Estimated linear regression parameters. A vector of shape (n+1, ).\n",
    "    \n",
    "    Instructions\n",
    "    ------------\n",
    "    Complete the code to compute the closed form solution to linear\n",
    "    regression and put the result in theta.\n",
    "    \n",
    "    Hint\n",
    "    ----\n",
    "    Look up the function `np.linalg.pinv` for computing matrix inverse.\n",
    "    \"\"\"\n",
    "    theta = np.zeros(X.shape[1])\n",
    "    \n",
    "    # ===================== YOUR CODE HERE ============================\n",
    "    theta=(np.linalg.pinv(X.T@X))@(X.T)@y\n",
    "\n",
    "    \n",
    "    # =================================================================\n",
    "    return theta"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "*You should now submit your solutions.*"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "Submitting Solutions | Programming Exercise linear-regression\n",
      "\n",
      "Use token from last successful submission (k.kaushik25@gmail.com)? (Y/n): y\n",
      "                                  Part Name |     Score | Feedback\n",
      "                                  --------- |     ----- | --------\n",
      "                           Warm up exercise |  10 /  10 | Nice work!\n",
      "          Computing Cost (for one variable) |  40 /  40 | Nice work!\n",
      "        Gradient Descent (for one variable) |  50 /  50 | Nice work!\n",
      "                      Feature Normalization |   0 /   0 | Nice work!\n",
      "    Computing Cost (for multiple variables) |   0 /   0 | Nice work!\n",
      "  Gradient Descent (for multiple variables) |   0 /   0 | Nice work!\n",
      "                           Normal Equations |   0 /   0 | Nice work!\n",
      "                                  --------------------------------\n",
      "                                            | 100 / 100 |  \n",
      "\n"
     ]
    }
   ],
   "source": [
    "grader[7] = normalEqn\n",
    "grader.grade()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Optional (ungraded) exercise: Now, once you have found $\\theta$ using this\n",
    "method, use it to make a price prediction for a 1650-square-foot house with\n",
    "3 bedrooms. You should find that gives the same predicted price as the value\n",
    "you obtained using the model fit with gradient descent (in Section 3.2.1)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Theta computed from the normal equations: [-3.89578088  1.19303364]\n",
      "Predicted price of a 1650 sq-ft, 3 br house (using normal equations): $0\n"
     ]
    }
   ],
   "source": [
    "# Calculate the parameters from the normal equation\n",
    "theta = normalEqn(X, y);\n",
    "\n",
    "# Display normal equation's result\n",
    "print('Theta computed from the normal equations: {:s}'.format(str(theta)));\n",
    "\n",
    "# Estimate the price of a 1650 sq-ft, 3 br house\n",
    "# ====================== YOUR CODE HERE ======================\n",
    "\n",
    "price = 0 # You should change this\n",
    "\n",
    "# ============================================================\n",
    "\n",
    "print('Predicted price of a 1650 sq-ft, 3 br house (using normal equations): ${:.0f}'.format(price))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
