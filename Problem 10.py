'''
 Runge-Kutta Method
====================================================================
Author: Avijit Maity
====================================================================
This Program solves the differential equation using
 with Runge-Kutta Method
'''

from numpy import *
from matplotlib.pyplot import  *

# The ordinary differential equation
# Here we will change our variable from t to u
# Put t = tan u

def g(x,u):
    return 1/(x**2+(tan(u))**2)

# Define the initial values and mesh
# limits: 0.0 <= t <= infinty
#limits: 0.0 <= u <= pi/2

a = 0.0
b = pi/2
h = 0.01
n = (b-a)/h

upoints = arange(a,b,h) # Create mesh
tpoints = []
xpoints = []

x = 1.0 # Initial value of x

for u in upoints: # Apply Runge-Kutta Method
    tpoints.append(tan(u))
    xpoints.append(x)
    k1 = h*g(x,u)
    k2 = h*g(x+0.5*k1,u+0.5*h)
    k3 = h*g(x+0.5*k2,u+0.5*h)
    k4 = h*g(x+k3,u+h)
    x += (k1+2*k2+2*k3+k4)/6



plot(tpoints,xpoints)
xlim(0,80)
title( "Runge-Kutta Method in a Python, h= 0.01" )
suptitle("Problem 10")
xlabel('t')
ylabel('x(t)')
legend(loc=4)
grid()
show()