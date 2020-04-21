'''
BVP SOLVING BY SHOOTING METHOD
====================================================================
Author: Avijit Maity
====================================================================
This Program solves the given Boundary value problem using
shooting method

Given equation

x''(t) = -g with boundary values

x(0) = x(t1) = 0

thus the vectorized equation for IVP  is:
let u = x v = x'
r' = f(r,t)

where r(t) = (u,v), r'(t) = (u',v') f(r,t) =(v,-g)

with initial guess r(0) = (0,s)

s being the iniial guess

'''
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op

g = 9.8
t1 = 0 # initial time
tf = 20  # Taken higher than t2
t2 = 10 # final time
x1 = 0.0  # initial position
x2 = 0.0 # final position
error = 1e-5


def x(t):
    return 0.5 * g * (t * t2 - t ** 2.0)


def f(r, t):  # The vector Function
    return np.array([r[1], -g])


def Euler_Solve(s): # This fun
    h = 0.01
    t = np.arange(t1, tf, h, dtype=np.float64)
    r = np.zeros(shape=(len(t), 2), dtype=np.float64)
    r[0] = np.array([x1, s])
    k = 0
    for i in range(len(t) - 1): # apply Runge kutta method
        k1 = h * f(r[i],t[i])
        k2 = h * f(r[i] + k1 / 2.0, t[i] + h / 2.0)
        k3 = h * f(r[i] + k2 / 2.0, t[i] + h / 2.0)
        k4 = h * f(r[i] + k3, t[i] + h)
        # Combine partial steps.
        r[i+1] = r[i ] + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    t_coord = int((t2 - t1) / h)
    return [r[t_coord, 0],
            [t[0:1 + np.argmin(np.abs(r[1:len(t) - 1, 0]))], r[0:1 + np.argmin(np.abs(r[1:len(t) - 1, 0])), 0]]]


def func(s):  # the function on which root solving is to be done
    return Euler_Solve(s)[0] - x2


root = op.newton(func, 2.0)

p = np.arange(root - 10.0, root + 10.0, 0.5, dtype=np.float64)
plt.title(" Solution by shooting method", size=18)
plt.xlabel("t")
plt.ylabel("x(t)")
plt.grid()
for i in range(len(p)):
    ar = Euler_Solve(p[i])[1]
    if (np.abs(p[i] - root) <= 0.001):
        plt.plot(ar[0], ar[1], color='#00F000', lw=5, label='True Numerical solution')
    elif (i == 0):
        plt.plot(ar[0], ar[1], ':', color='red', label='Trail solutions')
    else:
        plt.plot(ar[0], ar[1], ':', color='red')
plt.plot(np.linspace(0, t2, 1000), x(np.linspace(0, t2, 1000)), color='#000000', label="Analytical solution")
plt.legend()
plt.show()
