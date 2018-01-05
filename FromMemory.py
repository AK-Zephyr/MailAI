# -*- coding: utf-8 -*-
"""Email Neural Net

This program will take an input of arrays and and perform supervized learning
to predict against a known output.

Example:
    Import parsed email data and output a suggested responce.

Attributes:
    module_level_variable1 (int): Module level variables may be documented in
        either the ``Attributes`` section of the module docstring, or in an
        inline docstring immediately following the variable.

        Either form is acceptable, but the two should not be mixed. Choose
        one convention to document module level variables and be consistent
        with it.

Todo:
    * For module TODOs
    * You have to also use ``sphinx.ext.todo`` extension

.. _Google Python Style Guide:
   http://google.github.io/styleguide/pyguide.html

"""
import numpy as np



###########
##Variables
###########
#define seed for consistent results
np.random.seed(1)
#numpy array
input_array = [[0,0,1],
                [0,1,1],
                [1,0,1],
                [1,1,1]]
#numpy array -- will be transposed
output_array = [[0,0,1,1]]

#how many iternations
cycle_count = 10000
#log per how many cycles
log_amount = 100



###########
##Classes
###########
class Logging:
    def __init__(self, name):
        self.name = name
        self.document = open(str(self.name) + ".txt", "w")
    def write(self, msg):
        self.document.write(str(msg) + "\n")
    def close(self):
        self.document.close()



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
##Logging
###########
#set up instance of each output logs
error_correction = Logging(name="error_correction")
seed = Logging(name="seed")



###########
##Variable Setup
###########
# input dataset
data = np.array(input_array)
# output dataset
goal = np.array(output_array).T



###########
##Logic
###########
#initialize weights randomly with mean 0
array_seed = 2*np.random.random((3,1)) - 1

seed.write("Array Seed")
seed.write(array_seed)

for iter in range(cycle_count):
    # forward propagation
    knowledge = array_seed
    l0 = data
    l1 = nonlin(np.dot(l0,knowledge))
    # how much did we miss?
    l1_error = goal - l1
    # multiply how much we missed by the
    # slope of the sigmoin at the value in l1
    l1_delta = l1_error * nonlin(l1,True)
    # update weights
    knowledge += np.dot(l0.T,l1_delta)

    #Logging
    if not (iter % log_amount):
        error_correction.write("Cycle Count: " + str(iter))
        error_correction.write(l1_error)
        error_correction.write("")


#close logs
error_correction.close()
seed.close()
#output to screen
print("Output After Training")
print(l1)
