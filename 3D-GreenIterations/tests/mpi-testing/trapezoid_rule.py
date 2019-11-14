from mpi4py import MPI
import numpy as np


def f(x):
    return x*x

def Trap(a,b,n,h):
    integral = ( f(a) + f(b) ) / 2
    
    x = a
    for i in range(1, int(n)):
        x = x + h
        integral = integral + f(x)
        
    return integral * h

if __name__=="__main__":
    a = 0
    b = 1
    n = 100000
    h = (b-a)/n
    val = Trap(a,b,n,h) 
    print(val)