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
    return 1 - (t-y)**2

#  Define time spans, initial values, and constants
t = np.linspace(2, 3, 1000)
y0 = 0


# Solve differential equation
s= solve_ivp( f,[2,3],[0],dense_output="True")

#Exact soution
def exact(t):
    return (2- 5*t - 2*t**2)/(2*t - 5)

plot(t,s.sol(t).T, label='ivp')
plot( t, exact(t),':', color='red', label='exact' )
title( "Solving initial value problem using scipy.integrate.solve_ivp" )
suptitle("Problem 7b")
xlabel('t')
ylabel('y(t)')
legend(loc=4)
grid()
show()
