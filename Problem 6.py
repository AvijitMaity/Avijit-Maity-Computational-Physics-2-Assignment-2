'''

BVP SOLVING BY RELAXATION METHOD
====================================================================
Author: Avijit Maity
====================================================================
This Program solves the given Boundary value problem using
relaxation method

'''

from numpy import *
import matplotlib.pyplot as plt

h=0.2 # step size
t=arange(0,10+h,h) # Creating mesh points
g=-9.8
a=0 # Initial position
b=0 # Final position
N=len(t)-2
A=[[0 for i in range (N)]for j in range(N)]
B=zeros(N)
for i in range(N):
    A[i][i]=-2
    for j in range(N):
        if(j==i-1):
            A[i][j]=1
        elif(j==i+1):
            A[i][j]=1
    B[i]=g*(h**2)

B[0]=(g-a)*(h**2)
B[len(t)-3]=(g-b)*(h**2)


def gauss_seidel(x):
    p=copy(x)
    x=zeros(N)
    for j in range(len(x)):

       for k in range(len(x)):
           if k<j :
               x[j]=x[j]-(A[j][k]*x[k])
           if k>j:
                x[j]=x[j]-(A[j][k]*p[k])


       x[j]=(x[j]+B[j])/A[j][j]
    return x

X=[]
x=zeros(N)
Flag=1
c=0
while(Flag==1):
    z=zeros(len(t))
    for j in range(len(t)-2):
        z[j+1]=x[j]
    z[0]=0
    z[len(t)-1]=0
    X.append(z)
    p=copy(x)
    x=gauss_seidel(x)
    if(all(abs(x-p))<0.1):
        break

    c+=1


plt.plot(t,X[c])
plt.plot(t,X[int(c/10)])
plt.plot(t,X[int(c/15)])
plt.plot(t,X[int(c/20)])
plt.plot(t,X[int(c/25)])
plt.xlabel("x",size=18)
plt.ylabel("t",size=18)
plt.title("Problem 6",size=18)
plt.legend(loc=4)
plt.show()