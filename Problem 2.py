'''
 Euler’s Method
====================================================================
Author: Avijit Maity
====================================================================
This Program solves the given initial  value problem using
 with Euler’s Method
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


# The ordinary differential equation
def f(t,y ):
    return (y/t)- (y/t)**2

# Define the initial values and mesh
# limits: 1.0 <= t <= 2.0
a=1
b=2
h= 0.1 # determine step-size
t = np.arange( a, b+h, h ) # create mesh
N = int((b-a)/h)
y = np.zeros((N+1,)) # initialize y
t[0]=1
y[0]=1

for i in range(1,N+1): # apply Euler's method
    y[i] = y[i-1] + h * f( t[i-1], y[i-1] )


# The exact solution for this initial value
def exact( t ):
      return t/(1+np.log(t))

#We can use the following expression to evaluate the absolute
#error, which is the sum of the absolute values of the residuals
error_abs = lambda x, y: np.sum( np.abs( x - y ) )
print("The absolute error in the solution obtained using Euler’s method",error_abs( exact(t), y ))

#We can use the following expression to evaluate the relative
#error, which is the sum of the absolute values of the residuals
error_rel = lambda x, y: np.sum( np.abs( x - y ) / np.abs( x ) )
print("The relative error in the solution obtained using Euler’s method",error_abs( exact(t), y ))

# Plot the Euler solution
from matplotlib.pyplot import *
plot( t, y, label='approximation' )
plot( t, exact(t),':', color='red', label='exact' )
title( "Euler's Method Example, h= 0.1" )
suptitle("Problem 2")
xlabel('t')
ylabel('y(t)')
legend(loc=4)
grid()
show()

