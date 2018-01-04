#This Script is to test my ability to recall the basic functions of a NN from memory.
###########
##Imports
###########
import numpy as np
###########
##Outputs
###########
log = open("log.txt","w")
###########
##Variables
###########
np.random.seed(1)
input_array = [[0,0,1],[0,1,1],[1,0,1],[1,1,1]]
output_array = [[0,0,1,1]]
cycle_count = 10000
#log per how many cycles
log_amount = 100
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
print("Array Seed")
print (array_seed)
for iter in range(cycle_count):
    # forward propagation
    knowledge = array_seed
    l0 = data
    l1 = nonlin(np.dot(l0,array_seed))
    # how much did we miss?
    l1_error = Y - l1
    # multiply how much we missed by the
    # slope of the sigmoin at the value in l1
    l1_delta = l1_error * nonlin(l1,True)
    # update weights
    knowledge += np.dot(l0.T,l1_delta)

    # Logging
    if not (iter % log_amount):
        log.writelines(str(iter) + "\n")
        log.writelines(str(l1_error) + "\n")


print("Output After Training")
print(l1)
