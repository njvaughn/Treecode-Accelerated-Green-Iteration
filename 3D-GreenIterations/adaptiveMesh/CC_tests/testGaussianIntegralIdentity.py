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
    print('\ntmax =  ', tmax)
    print('dt =    ', tvec[1]-tvec[0])
    print('1/r =   ', 1/r)
    
    
    I = 0.0
    for i in range(timeIntervals):
        # perform a midpoint method in t
        dt = tvec[i+1]-tvec[i]
        t = (tvec[i+1]+tvec[i])/2
        I += np.exp(-t**2 * r**2)*dt
        
    I *= 2/np.sqrt(np.pi)
    
    print('I =     ', I)
    print('error = ', I-1/r)
    print()
        
if __name__ == "__main__":
    x = 1e-3
    y = 0
    z = 0
    
#     ratio = 0.5

#     timesteps = 40
#     tmax = ratio/(np.sqrt(x**2 + y**2 + z**2))


    tmax = 3000
    dt = 50
    timesteps = int( np.ceil( tmax / dt) )
    
    
    
    gaussianIntegral(x,y,z,0,0,0,tmax,timesteps)
    
    x = 1e-2
    y = 0
    z = 0
    gaussianIntegral(x,y,z,0,0,0,tmax,timesteps)
    
    
#     gaussianIntegral(x,y,z,0,0,0,ratio/(np.sqrt(x**2 + y**2 + z**2)),timesteps)
    
    