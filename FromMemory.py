#This Script is to test my ability to recall the basic functions of a NN from memory.

import numpy as np

##Variables
np.random.seed(1)
InputArray = [[0,0,1],[0,1,1],[1,0,1],[1,1,1]]

OutputArray = [[0,0,1,1]]

CycleCount = 10000
np.random.seed(1)

##Setup
#sigmoid function
def nonlin(x, deriv=False):
        if(deriv==True):
            return x*(1-x)
        return 1/(1+np.exp(-x))

# input dataset
X = np.array(InputArray)

# output dataset
Y = np.array(OutputArray.T)

##Logic
#initialize weights randomly with mean 0
syn0 = 2*np.random.random((3,1)) - 1

for iter in xrange(CycleCount):

    # forward propagation
    l0 = X
    l1 = nonlin(np.dot(l0,syn0))
