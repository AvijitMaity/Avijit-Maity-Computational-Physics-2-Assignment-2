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
from matplotlib.pyplot import *
def f(t,y):
    return np.cos(2*t) + np.sin(3*t)

#  Define time spans, initial values, and constants
t = np.linspace(0, 1, 1000)



# %% Solve differential equation
s= solve_ivp( f,[0,1],[1],dense_output="True")

#Exact soution
def exact(t):
    return (8- 2*np.cos(3*t) + 3*np.sin(2*t))/6

plot(t,s.sol(t).T, label='ivp')
plot( t, exact(t),':', color='red', label='exact' )
title( "Solving initial value problem using scipy.integrate.solve_ivp" )
suptitle("Problem 7d")
xlabel('t')
ylabel('y(t)')
legend(loc=4)
grid()
show()
