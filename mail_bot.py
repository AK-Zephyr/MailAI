# -*- coding: utf-8 -*-
"""Email Neural Net

This program will take an input of arrays and and perform supervized learning
to predict against a known output.

Example:
    Import parsed email data and output a suggested responce.

Attributes:
    INPUT_ARRAY (array): Input an array formatted for numpy.array.
    The program will automatically scale based on the size of the input.

    OUTPUT_ARRAY (array): The known or suggested outcome of the supervised learning.

Todo:
    * Add Input/Output scaling
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
INPUT_ARRAY = [[0, 0, 1],
               [0, 1, 1],
               [1, 0, 1],
               [1, 1, 1]]
#numpy array -- will be transposed
OUTPUT_ARRAY = [[0,
                 0,
                 1,
                 1]]

#how many iternations
CYCLE_COUNT = 10000
#log per how many cycles
LOG_AMOUNT = 100



###########
##Classes
###########
class Logging:
    """Create a txt document with write default.

    Attributes:
            name (str): Name of log file.
    """
    def __init__(self, name):
        """Initializes file object with 'w' permissions.
        """
        self.name = name
        self.document = open(self.name, "w")

    def write(self, msg):
        """
        """
        self.document.write(str(msg) + "\n")
    def close(self):
        self.document.close()



###########
##Functions
###########
def nonlin(x, deriv=False):

    This is a sigmoid function
    (input,True=return derivative)
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))



###########
##Logging
###########
#set up instance of each output logs
error_correction = Logging(name="error_correction.txt")
seed = Logging(name="seed.txt")



###########
##Variable Setup
###########
# input dataset
data = np.array(INPUT_ARRAY)
# output dataset
goal = np.array(OUTPUT_ARRAY).T



###########
##Logic
###########
#initialize weights randomly with mean 0
array_seed = 2*np.random.random((3,1)) - 1

seed.write("Array Seed")
seed.write(array_seed)

for step in range(CYCLE_COUNT):
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
    if not (step % LOG_AMOUNT):
        error_correction.write("Cycle Count: " + str(iter))
        error_correction.write(l1_error)
        error_correction.write("")


#close logs
error_correction.close()
seed.close()
#output to screen
print("Output After Training")
print(l1)
