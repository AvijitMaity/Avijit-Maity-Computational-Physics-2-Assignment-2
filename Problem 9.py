'''
IVP SOLVING BY ADAPTIVE STEP SIZE  METHOD
====================================================================
Author: Avijit Maity
====================================================================
This Program solves the given Boundary value problem using
adaptive step size method

Given first order equation:
y' = (y^2+y)/t = f(y,t)
with 1<=t<=3 and y(1) = -2

Adaptive step size relies on the test condition:
if e is the permisible error and and if y1 and y2 are the values of the
fuction at some t+2h such that for y1 it was obtained via
t --> t+h --> t+2h and for y2 it was obtained via t ---> t+2h, where h is
the current step size. Then the optimal stepsize to obtain an error bound e is

h' = (30.0*h*e/(|y1-y2|))^0.25 * h

thus for any necessary step, this is the optimal step size.
We had already executed RK4 Schemes in previous sections and will be
using the same algorithms to solve these.
'''

import numpy as np
from scipy.optimize import *
import matplotlib.pyplot as plt
from scipy.integrate import *


def f(y, t):  # defining function that returns derivative

    return (y ** 2 + y) / t


def RK(h, t, w):  # function for 4th order Runge-Kutta method
    k1 = h * f(w, t)
    k2 = h * f(w + k1 / 2, t + h / 2)
    k3 = h * f(w + k2 / 2, t + h / 2)
    k4 = h * f(w + k3, t + h)
    w = w + 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return w


d = 0.0001  # accuracy
a = 1  # initial poni
b = 3  # final point
h = 0.01  # initial step size
t = [a]  # creating array that stores the t value
w = [-2]  # initial value of y is -2
Flag = 1
while (Flag == 1):

    y = RK(h, t[len(t) - 1], w[len(w) - 1])
    y1 = RK(h, t[len(t) - 1] + h, y)  #  the value of the  function at t_i + 2h using RK4 twice with step size h

    y2 = RK(2 * h, t[len(t) - 1], w[len(w) - 1])  #  the value of the  function at t_i + 2h using RK4 once with step size 2h

    h = h * (d * h * 30 / abs(y1 - y2)) ** 0.25  # calculating the perfect step size

    if (t[len(t) - 1] + h > b):

        h = b - t[len(t) - 1]
        y = RK(h, t[len(t) - 1], w[len(w) - 1])  # calculating the value of y with the perfect step size
        t.append(b)  # appending the t values
        w.append(y)  # appending the y values
        break
    y = RK(h, t[len(t) - 1], w[len(w) - 1])  # calculating the value of y with the perfect step size

    t.append(t[len(t) - 1] + h)  # appending the t values
    w.append(y)  # appending the y values

# Here we will solve this ODE using odient
#we change the variable t to x
# function that returns dy/dx
def model(y,x):
    dydx = (y ** 2 + y) / x
    return dydx

# initial condition
y0 = -2
x = np.arange( a, b+0.01, 0.01 ) # h=0.01

# solve ODE
Sol = odeint(model,y0,x)


plt.scatter(t, w)
plt.plot( x, Sol, color='red', label='odient' )
plt.legend(loc=4)
plt.xlabel("t", size=14)
plt.ylabel("y", size=14)
plt.title( "Adaptive step size  Method in a Python code" )
plt.suptitle("Problem 9")
plt.show()