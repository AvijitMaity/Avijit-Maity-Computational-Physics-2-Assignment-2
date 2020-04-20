'''
 Runge-Kutta Method
====================================================================
Author: Avijit Maity
====================================================================
This Program solves the initial value problem using
 with Runge-Kutta Method
'''


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


# The ordinary differential equation
def f1(t,u1,u2,u3 ):
    return u1 + 2*u2 - 2*u3 + np.exp(-t)

def f2(t,u1,u2,u3):
    return u2 + u3 -2*np.exp(-t)

def f3(t,u1,u2,u3):
    return u1 + 2*u2 + np.exp(-t)


# Define the initial values and mesh
# limits: 0.0 <= t <= 1.0
a=0
b=1
h= 0.01 # determine step-size
t = np.arange( a, b+h, h ) # create mesh
N = int((b-a)/h)
u1 = np.zeros((N+1,)) # initialize u1
u2 = np.zeros((N+1,)) # initialize u2
u3 = np.zeros((N+1,)) # initialize u2

t[0]=0
u1[0]=3 # Initial value of u1
u2[0]=-1 # Initial value of u2
u3[0]=1 # Initial value of u3


for i in range(1,N+1): # Apply Runge-Kutta Method
    k1 = h * f1(t[i - 1], u1[i - 1], u2[i - 1], u3[i - 1])
    l1 = h * f2(t[i - 1], u1[i - 1], u2[i - 1], u3[i - 1])
    m1 = h * f3(t[i - 1], u1[i - 1], u2[i - 1], u3[i - 1])

    k2 = h * f1(t[i - 1] + h / 2.0, u1[i - 1] + k1 / 2.0, u2[i - 1] + l1 / 2.0 ,u3[i - 1] + m1 / 2.0)
    l2 = h * f2(t[i - 1] + h / 2.0, u1[i - 1] + k1 / 2.0, u2[i - 1] + l1 / 2.0, u3[i - 1] + m1 / 2.0)
    m2 = h * f3(t[i - 1] + h / 2.0, u1[i - 1] + k1 / 2.0, u2[i - 1] + l1 / 2.0, u3[i - 1] + m1 / 2.0)

    k3 = h * f1(t[i - 1] + h / 2.0, u1[i - 1] + k2 / 2.0, u2[i - 1] + l2 / 2.0, u3[i - 1] + m2 / 2.0)
    l3 = h * f2(t[i - 1] + h / 2.0, u1[i - 1] + k2 / 2.0, u2[i - 1] + l2 / 2.0, u3[i - 1] + m2 / 2.0)
    m3 = h * f3(t[i - 1] + h / 2.0, u1[i - 1] + k2 / 2.0, u2[i - 1] + l2 / 2.0, u3[i - 1] + m2 / 2.0)

    k4 = h * f1(t[i-1]+h, u1[i - 1] + k3, u2[i - 1] + l3, u3[i - 1] + m3)
    l4 = h * f2(t[i-1]+h, u1[i - 1] + k3, u2[i - 1] + l3, u3[i - 1] + m3)
    m4 = h * f3(t[i - 1] + h, u1[i - 1] + k3, u2[i - 1] + l3, u3[i - 1] + m3)

    u1[i] = u1[i - 1] + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    u2[i] = u2[i - 1] + (l1 + 2.0 * l2 + 2.0 * l3 + l4) / 6.0
    u3[i] = u3[i - 1] + (m1 + 2.0 * m2 + 2.0 * m3 + m4) / 6.0



# Plot the Euler solution
from matplotlib.pyplot import *
plot( t, u1, label='u1 vs t' )
plot( t, u2, label='u2 vs t' )
plot( t, u3, label='u3 vs t' )

title( "Runge-Kutta Method in a Python, h= 0.01" )
suptitle("Problem 11")
xlabel('x')
ylabel('y(x)')
legend(loc=4)
grid()
show()