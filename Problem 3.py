'''
 Runge-Kutta Method
====================================================================
Author: Avijit Maity
====================================================================
This Program solves the differential equation using
 with Runge-Kutta Method
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


# The ordinary differential equation
def f(x,y,z ):
    return z

def g(x,y,z ):
    return 2*z - y + x*np.exp(x) - x

# Define the initial values and mesh
# limits: 0.0 <= x <= 1.0
a=0
b=1
h= 0.01 # determine step-size
x = np.arange( a, b+h, h ) # create mesh
N = int((b-a)/h)
y = np.zeros((N+1,)) # initialize y
z = np.zeros((N+1,)) # initialize z

x[0]=0
y[0]=0
z[0]=0


for i in range(1,N+1): # Apply Runge-Kutta Method

    k1 = h * f(x[i - 1], y[i - 1], z[i - 1])
    l1 = h * g(x[i - 1], y[i - 1], z[i - 1])

    k2 = h * f(x[i - 1] + h / 2.0, y[i - 1] + k1 / 2.0, z[i - 1] + l1 / 2.0 )
    l2 = h * g(x[i - 1] + h / 2.0, y[i - 1] + k1 / 2.0, z[i - 1] + l1 / 2.0 )

    k3 = h * f(x[i - 1] + h / 2.0, y[i - 1] + k2 / 2.0, z[i - 1] + l2 / 2.0)
    l3 = h * g(x[i - 1] + h / 2.0, y[i - 1] + k2 / 2.0, z[i - 1] + l2 / 2.0)

    k4 = h * f(x[i-1]+h, y[i - 1] + k3, z[i - 1] + l3)
    l4 = h * g(x[i-1]+h, y[i - 1] + k3, z[i - 1] + l3)

    y[i] = y[i - 1] + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    z[i] = z[i - 1] + (l1 + 2.0 * l2 + 2.0 * l3 + l4) / 6.0


# The exact solution for this initial value
def exact( x ):
      return (-12 + 12*np.exp(x) - 6*x - 6*np.exp(x)*x + np.exp(x)*x**3)/6

# Plot the Euler solution
from matplotlib.pyplot import *
plot( x, y, label='approximation' )
plot( x, exact(x),":",color="red",label='exact' )
title( "Runge-Kutta Method in a Python, h= 0.01" )
suptitle("Problem 3")
xlabel('x')
ylabel('y(x)')
legend(loc=4)
grid()
show()
