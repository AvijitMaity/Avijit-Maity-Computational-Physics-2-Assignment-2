'''
Backward integration with Euler’s Method
====================================================================
Author: Avijit Maity
====================================================================
This Program solves the given initial  value problem using
backward integration with Euler’s Method
'''


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# The ordinary differential equation
def f(x,y ):
    return  -20*(y-x)**2+ 2*x

# Define the initial values and mesh
# limits: 0.0 <= x <= 1.0
a=0
b=1
h= 0.01 # determine step-size
x = np.arange( a, b+h, h ) # create mesh
N = int((b-a)/h)
y = np.zeros((N+1,)) # initialize y
x[0]=0
y[0]=1/3

for i in range(1,N+1): # apply backward integration with Euler’s Method
    ye= y[i-1] + h * f( x[i-1], y[i-1] )
    y[i] = y[i-1] + h * f( x[i], ye )

# function that returns dy/dx
def model(y,x):
    dydx = -20*(y-x)**2+ 2*x
    return dydx

# initial condition
y0 = 1/3

# solve ODE
Sol = odeint(model,y0,x)

# Plot the Euler solution
from matplotlib.pyplot import *
plot( x, y, label='approximation' )
plot( x, Sol, label='odient' )
title( "Backward integration with Euler’s Method in a Python, h= 0.01" )
suptitle("Problem 1b")
xlabel('x')
ylabel('y(x)')
legend(loc=4)
grid()
show()
