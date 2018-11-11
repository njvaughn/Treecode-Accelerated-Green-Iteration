'''
Created on June 7, 2018

@author: nathanvaughn
'''
import unittest
import numpy as np
from numpy.random import randint
import itertools


def gaussianIntegral(xt,yt,zt,xs,ys,zs,tmax,timeIntervals):
    
    tvec = np.linspace(0,tmax,timeIntervals+1)
#     print(tvec)
    
    r = np.sqrt( (xt-xs)**2 + (ys-yt)**2 + (zt-zs)**2 )
    print('\ntmax = ', tmax)
    print('dt =   ', tvec[1]-tvec[0])
    print('1/r =  ', 1/r)
    
    I = 0.0
    for i in range(timeIntervals):
        # perform a midpoint method in t
        dt = tvec[i+1]-tvec[i]
        t = (tvec[i+1]+tvec[i])/2
        I += np.exp(-t**2 * r**2)*dt
        
    I *= 2/np.sqrt(np.pi)
    
    print('I =    ', I)
    print()
        
if __name__ == "__main__":
    gaussianIntegral(1e-1,0,0,0,0,0,5e4,20)
    gaussianIntegral(1e-1,0,0,0,0,0,5e4,100)
    gaussianIntegral(1e-1,0,0,0,0,0,5e4,5000)
    