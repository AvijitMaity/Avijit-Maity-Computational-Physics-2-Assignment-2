'''
Solving initial value problems
using scipy.integrate.solve_bvp
====================================================================
Author: Avijit Maity
====================================================================
This Program solves initial value problems
using scipy.integrate.solve_bvp
'''

import numpy as np
from scipy.integrate import *
import matplotlib.pyplot as plt
from numpy import *

a=0         #starting value
b=(np.pi)/2       #end value

#We rewrite the equation as a first order system and
# #implement its right-hand side evaluation

def fun(x,y):
    return np.vstack((y[1],y[1]*(np.cos(x))-y[0]*(np.log(y[0]))))

#Implement evaluation of the boundary condition residuals:
def bc(ya,yb):               #function for boundary conditions
    return np.array([ya[0]-1,yb[0]-np.exp(1)])

x=np.linspace(a,b,100)       #creating mesh points
y=np.zeros((2,x.size))
y[0]=1            #initial guess of the solution

sol=solve_bvp(fun,bc,x,y)     #solving the problem using solve_bvp

y_NDSolve=np.loadtxt("txt file of problem 8b.txt",usecols=0)     #importing the text file containing the solution using Mathematica
plt.plot(x,y_NDSolve,':',color='red',label="analytical solution using mathematica")         #it will plott the analytical solution using mathematica
plt.plot(x,sol.sol(x)[0],color='yellow',label="solve_bvp")    #plotting the solution using solve_bvp
plt.legend(loc=4)
plt.xlabel("x",size=18)
plt.ylabel("y",size=18)
plt.title( "Solving initial value problem using scipy.integrate.solve_bvp" )
plt.suptitle("Problem 8b")
plt.grid()
plt.show()