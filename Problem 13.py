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
def f(t,y,z):
    return z

def g(t,y,z):
    return 2*(z/t) - 2*(y/t**2)+ t*np.log(t)

# Define the initial values and mesh
# limits: 1.0 <= t <= 2.0
a=1
b=2
h= 0.001 # determine step-size
t = np.arange( a, b+h, h ) # create mesh
N = int((b-a)/h)
y = np.zeros((N+1,)) # initialize y
z = np.zeros((N+1,)) # initialize z

t[0]=1
y[0]=1 # initial value of y
z[0]=0 # initial value of dy/dt

for i in range(1,N+1): # apply Euler's method
    y[i] = y[i-1] + h * f( t[i-1], y[i-1], z[i-1])
    z[i] = z[i - 1] + h * g(t[i - 1], y[i - 1], z[i - 1])


# The exact solution for this initial value
def exact( t ):
      return (7/4)*t + (t**3 * np.log(t))/2 - (3/4)*t**3


# Plot the Euler solution
from matplotlib.pyplot import *
plot( t, y, label='approximation' )
plot( t, exact(t),':', color='red', label='exact' )
title( "Euler's Method Example, h= 0.001" )
suptitle("Problem 13")
xlabel('t')
ylabel('y(t)')
legend(loc=4)
grid()
show()