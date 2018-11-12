"""
A CUDA version of the convolution for Green Iterations.
It is written so that each thread is responsible for one of the N target gridpoints.  
Each thread interacts with all M source midpoints. 
Note: the current implementation does not use shared memory, which would provide additional
speedup.  -- 03/19/2018 NV 
"""
from numba import cuda
from math import sqrt,exp,factorial,pi, erfc
import math
import numpy as np
import os
import shutil
from timeit import default_timer as timer
from numpy import float32, float64, int32





@cuda.jit('void(float64[:,:], float64[:,:], float64[:], float64)')
def gpuHelmholtzConvolution_skip_generic_selfCell(targets,sources,psiNew,k):
    globalID = cuda.grid(1)  
    if globalID < len(targets):  
        x_t, y_t, z_t, f_t, weight_t, initval_t, id_t = targets[globalID] 
        psiNew[globalID] = initval_t
        for i in range(len(sources)): 
            x_s, y_s, z_s, f_s, weight_s, initval_s, id_s = sources[i]
            if id_t != id_s: 
                r = sqrt( (x_t-x_s)**2 + (y_t-y_s)**2 + (z_t-z_s)**2 ) 
                if r > 1e-14:
                    psiNew[globalID] += weight_s*f_s*exp(-k*r)/(r) 

@cuda.jit('void(float64[:,:], float64[:,:], float64[:], float64)')
def gpuHelmholtzConvolution_subtract_generic_selfCell(targets,sources,psiNew,k):
    globalID = cuda.grid(1)  
    if globalID < len(targets):  
        x_t, y_t, z_t, f_t, weight_t, initval_t, id_t = targets[globalID]  
        psiNew[globalID] = initval_t + 4*pi*f_t/k**2
        for i in range(len(sources)):  
            x_s, y_s, z_s, f_s, weight_s, initval_s, id_s = sources[i] 
            if id_t != id_s: 
                r = sqrt( (x_t-x_s)**2 + (y_t-y_s)**2 + (z_t-z_s)**2 ) 
                if r > 1e-14: 
                    psiNew[globalID] += weight_s*(f_s-f_t)*exp(-k*r)/(r)
                    
                    
@cuda.jit('void(float64[:,:], float64[:,:], float64[:])')
def gpuPoisson_selfCell(targets,sources,psiNew):
    globalID = cuda.grid(1)  
    if globalID < len(targets):  
        x_t, y_t, z_t, f_t, weight_t, initval_t, id_t = targets[globalID] 
        psiNew[globalID] = initval_t
        for i in range(len(sources)): 
            x_s, y_s, z_s, f_s, weight_s, initval_s, id_s = sources[i]
            if id_t != id_s: 
                r = sqrt( (x_t-x_s)**2 + (y_t-y_s)**2 + (z_t-z_s)**2 ) 
                if r > 1e-14:
                    psiNew[globalID] += weight_s*f_s/(r) 
                    
@cuda.jit('void(float64[:,:], float64[:,:], float64[:])')
def gpuPoisson_selfCell_gaussianSingularitySubtraction(targets,sources,psiNew,alpha):
    globalID = cuda.grid(1)  
    if globalID < len(targets):  
        x_t, y_t, z_t, f_t, weight_t, initval_t, id_t = targets[globalID] 
        psiNew[globalID] = initval_t
        for i in range(len(sources)): 
            x_s, y_s, z_s, f_s, weight_s, initval_s, id_s = sources[i]
            if id_t != id_s: 
                r = sqrt( (x_t-x_s)**2 + (y_t-y_s)**2 + (z_t-z_s)**2 ) 
                if r > 1e-14:
                    psiNew[globalID] += weight_s*(f_s-f_t*exp(- r*r / alphasq ) ) / r 

