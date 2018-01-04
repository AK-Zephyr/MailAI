#This Script is to test my ability to recall the basic functions of a NN from memory.
###########
##imports
###########
import numpy as np
###########
##Variables
###########
np.random.seed(1)
input_array = [[0,0,1],[0,1,1],[1,0,1],[1,1,1]]
output_array = [[0,0,1,1]]
cycle_count = 10000
###########
##Functions
###########
def nonlin(x, deriv=False):
    """This is a sigmoid function
    (input,True=return derivative)"""
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))
###########
##Variable Setup
###########
# input dataset
data = np.array(InputArray)
# output dataset
goal = np.array(OutputArray).T
###########
##Logic
###########
#initialize weights randomly with mean 0
array_seed = 2*np.random.random((3,1)) - 1

for iter in range(CycleCount):


    # forward propagation
    l0 = data
    l1 = nonlin(np.dot(l0,array_seed))

    # how much did we miss?
    l1_error = Y - l1
    # multiply how much we missed by the
    # slope of the sigmoin at the value in l1
    l1_delta = l1_error * nonlin(l1,True)

    # update weights
    array_seed += np.dot(l0.T,l1_delta)

print("Output After Training")
print(l1)
