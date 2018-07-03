"""
A CUDA version of the convolution for Green Iterations.
It is written so that each thread is responsible for one of the N target gridpoints.  
Each thread interacts with all M source midpoints. 
Note: the current implementation does not use shared memory, which would provide additional
speedup.  -- 03/19/2018 NV 
"""
from numba import cuda
from math import sqrt,exp,factorial,pi
import math
import numpy as np
import os
import shutil
from timeit import default_timer as timer
from numpy import float32

# from hydrogenPotential import trueEnergy, trueWavefunction
    
@cuda.jit('void(float64[:,:], float64[:,:], float64[:], float64)')
def gpuConvolution(targets,sources,psiNew,k):
    globalID = cuda.grid(1)  # identify the global ID of the thread
    if globalID < len(targets):  # check that this global ID doesn't excede the number of targets
        x_t, y_t, z_t = targets[globalID][0:3]  # set the x, y, and z values of the target
        for i in range(len(sources)):  # loop through all source midpoints
            x_s, y_s, z_s, psi_s, V_s, weight_s = sources[i]  # set the coordinates, psi value, external potential, and volume for this source cell
            if not ( (x_s==x_t) and (y_s==y_t) and (z_s==z_t) ):  # skip the convolutions when the target gridpoint = source midpoint, as G(r=r') is singular
#             if  globalID != i:  # skip the convolutions when the target gridpoint = source midpoint, as G(r=r') is singular
                r = sqrt( (x_t-x_s)**2 + (y_t-y_s)**2 + (z_t-z_s)**2 ) # compute the distance between target and source
                increment = 2*V_s*weight_s*psi_s*exp(-k*r)/r
#                 if abs(increment) < 1e1:
                psiNew[globalID] += increment # increment the new wavefunction value
#                 psiNew[globalID] = targets[globalID][3] # increment the new wavefunction value

def cpuConvolution(targets,sources,psiNew,k):
    for j in range(len(targets)):
        x_t, y_t, z_t = targets[j][0:3]
        for i in range(len(sources)):  # loop through all source midpoints
            x_s, y_s, z_s, psi_s, V_s, weight_s = sources[i][:]
            if i != j:
                r = sqrt( (x_t-x_s)**2 + (y_t-y_s)**2 + (z_t-z_s)**2 )
                increment = 2*V_s*weight_s*psi_s*exp(-k*r)/r
                if abs(increment) > 1e5:
                    print('increment = ', increment)
                    print('target = ', targets[j])
                    print('source = ', sources[i])
                    print('source x = ', x_s)
                    print('r = ',r)
                psiNew[j] += 2*V_s*weight_s*psi_s*exp(-k*r)/r
        print(psiNew[j])
    return psiNew

@cuda.jit
def gpuHelmholtzConvolutionSubractSingularity(targets,sources,psiNew,k):
    globalID = cuda.grid(1)  # identify the global ID of the thread
    if globalID < len(targets):  # check that this global ID doesn't excede the number of targets
        x_t, y_t, z_t, psi_t, V_t, weight_t = targets[globalID]  # set the x, y, and z values of the target
        f_t = 2*psi_t*V_t
#         psiNew[globalID] = f_t*4*pi/k**2
        psiNew[globalID] = f_t/k**2
        for i in range(len(sources)):  # loop through all source midpoints
            x_s, y_s, z_s, psi_s, V_s, weight_s = sources[i]  # set the coordinates, psi value, external potential, and volume for this source cell
            if not ( (x_s==x_t) and (y_s==y_t) and (z_s==z_t) ):  # skip the convolutions when the target gridpoint = source midpoint, as G(r=r') is singular
#             if  globalID != i:  # skip the convolutions when the target gridpoint = source midpoint, as G(r=r') is singular
                r = sqrt( (x_t-x_s)**2 + (y_t-y_s)**2 + (z_t-z_s)**2 ) # compute the distance between target and source
                f_s = 2*V_s*psi_s
                psiNew[globalID] += weight_s*(f_s-f_t)*exp(-k*r)/(4*pi*r) # increment the new wavefunction value

@cuda.jit
def gpuPoissonConvolution(targets,sources,V_coulomb_new):
    globalID = cuda.grid(1)  # identify the global ID of the thread
    if globalID < len(targets):  # check that this global ID doesn't excede the number of targets
        x_t, y_t, z_t = targets[globalID][0:3] # set the x, y, and z values of the target
        V_coulomb_new[globalID] = 0.0
        for i in range(len(sources)):  # loop through all source midpoints
            x_s, y_s, z_s, rho_s, weight_s = sources[i]  # set the coordinates, psi value, external potential, and volume for this source cell
#             if not ( abs(x_s-x_t) and (y_s==y_t) and (z_s==z_t) ):  # skip the convolutions when the target gridpoint = source midpoint, as G(r=r') is singular
            r = sqrt( (x_t-x_s)**2 + (y_t-y_s)**2 + (z_t-z_s)**2 ) # compute the distance between target and source
            if r > 1e-12:
                V_coulomb_new[globalID] += weight_s*rho_s/r # increment the new wavefunction value


               
@cuda.jit
def gpuConvolutionSmoothing(targets,sources,psiNew,k,n,epsilon,skipSingular=False):
    globalID = cuda.grid(1)  # identify the global ID of the thread
    if globalID < len(targets):  # check that this global ID doesn't excede the number of targets
        x_t, y_t, z_t = targets[globalID][0:3]  # set the x, y, and z values of the target
        for i in range(len(sources)):  # loop through all source midpoints
            x_s, y_s, z_s, psi_s, V_s, volume_s = sources[i]  # set the coordinates, psi value, external potential, and volume for this source cell
            r = sqrt( (x_t-x_s)**2 + (y_t-y_s)**2 + (z_t-z_s)**2 ) # compute the distance between target and source
            scaledEpsilon = epsilon * volume_s**(1/3)
#             scaledEpsilon = epsilon
            if ( (skipSingular==False) or (x_t!=x_s) or (y_t!=y_s) or (z_t!=z_s) ):
                
                ## COMPUTE CHRISTLIEB KERNEL APPROXIMATION ##
                Gvalue = 0.0
                for ii in range(n+1):
                    coefficient = 1.0
                    for jj in range(ii):
                        coefficient /= (jj+1) # this is replacing the 1/factorial(i)
                        coefficient *= ((-1/2)-jj)
                    
#                     Gvalue += coefficient* (-epsilon**2)**ii * (r**2 + epsilon**2)**(-1/2-ii)
                    Gvalue += coefficient* (-scaledEpsilon**2)**ii * (r**2 + scaledEpsilon**2)**(-1/2-ii)
    #             return -np.exp(-r)*Gvalue
                psiNew[globalID] += -2*V_s*volume_s*psi_s*exp(-k*r)*Gvalue # increment the new wavefunction value

# def dummyConvolutionToTestImportExport(targets,sources,psiNew,k):
#     for i in range(targets):
#         x_t, y_t, z_t = targets[i][0:3]  # set the x, y, and z values of the target
#         x_s, y_s, z_s, psi_s, V_s, volume_s = sources[i]
#         psiNew[i] = psi_s
#         
#         return psiNew
#         for i in range(len(sources)):  # loop through all source midpoints
#             x_s, y_s, z_s, psi_s, V_s, volume_s = sources[i]  # set the coordinates, psi value, external potential, and volume for this source cell
#             if not ( (x_s==x_t) and (y_s==y_t) and (z_s==z_t) ):  # skip the convolutions when the target gridpoint = source midpoint, as G(r=r') is singular
#                 r = sqrt( (x_t-x_s)**2 + (y_t-y_s)**2 + (z_t-z_s)**2 ) # compute the distance between target and source
#                 psiNew[globalID] += -2*V_s*volume_s*psi_s*exp(-k*r)/r # increment the new wavefunction value
 