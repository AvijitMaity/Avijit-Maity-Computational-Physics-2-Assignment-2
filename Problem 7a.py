'''
Solving initial value problems
using scipy.integrate.solve_ivp
====================================================================
Author: Avijit Maity
====================================================================
This Program solves initial value problems
using scipy.integrate.solve_ivp
'''


from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
def f(t,y):
    return t*np.exp(3*t) - 2*y

#  Define time spans, initial values, and constants
t = np.linspace(0, 1, 1000)
y0=0

# Solve differential equation
s= solve_ivp( f,[0,1],[0],dense_output="True")

#Exact soution
def exact(t):
    return (1/25)*(np.exp(-2*t))*(1-np.exp(5*t)+5*t*np.exp(5*t))

plot(t,s.sol(t).T, label='ivp')
plot( t, exact(t),":",color="red", label='exact' )
title( "Solving initial value problem using scipy.integrate.solve_ivp" )
suptitle("Problem 7a")
xlabel('t')
ylabel('y(t)')
legend(loc=4)
grid()
show()