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



# from hydrogenPotential import trueEnergy, trueWavefunction
    
@cuda.jit('void(float64[:,:], float64[:,:], float64[:], float64)')
def gpuHelmholtzConvolution(targets,sources,psiNew,k):
    globalID = cuda.grid(1)  # identify the global ID of the thread
    if globalID < len(targets):  # check that this global ID doesn't excede the number of targets
        x_t, y_t, z_t = targets[globalID][0:3]  # set the x, y, and z values of the target
        psiNew[globalID] = 0.0
        for i in range(len(sources)):  # loop through all source midpoints
            x_s, y_s, z_s, f_s, weight_s = sources[i]  # set the coordinates, psi value, external potential, and volume for this source cell
            r = sqrt( (x_t-x_s)**2 + (y_t-y_s)**2 + (z_t-z_s)**2 ) # compute the distance between target and source
            if (r > 1e-12):
                psiNew[globalID] += f_s*exp(-k*r)/(r)*weight_s #increment # increment the new wavefunction value
        psiNew[globalID] /= 4*pi
        
@cuda.jit('void(float64[:,:], float64[:,:], float64[:], float64)')
def gpuHelmholtzConvolutionSubractSingularity(targets,sources,psiNew,k):

    globalID = cuda.grid(1)  # identify the global ID of the thread
    if globalID < len(targets):  # check that this global ID doesn't excede the number of targets
        x_t, y_t, z_t, f_t = targets[globalID][0:4]  # set the x, y, and z values of the target
        psiNew[globalID] = 4*pi*f_t/k**2
        for i in range(len(sources)):  # loop through all source midpoints
            x_s, y_s, z_s, f_s, weight_s = sources[i]  # set the coordinates, psi value, external potential, and volume for this source cell
            r = sqrt( (x_t-x_s)**2 + (y_t-y_s)**2 + (z_t-z_s)**2 ) # compute the distance between target and source
            if (r > 1e-12 ):  # skip the convolutions when the target gridpoint = source midpoint, as G(r=r') is singular
                psiNew[globalID] += weight_s*(f_s-f_t)*exp(-k*r)/r # increment the new wavefunction value
        psiNew[globalID] /= 4*pi
        
        

@cuda.jit('void(float64[:,:], float64[:,:], float64[:], float64)')
def gpuHelmholtzConvolution_skip_generic(targets,sources,psiNew,k):
    globalID = cuda.grid(1)  
    if globalID < len(targets):  
        x_t, y_t, z_t = targets[globalID][0:3]  
        psiNew[globalID] = 0.0
        for i in range(len(sources)):  
            x_s, y_s, z_s, f_s, weight_s = sources[i] 
            r = sqrt( (x_t-x_s)**2 + (y_t-y_s)**2 + (z_t-z_s)**2 ) 
            if r > 1e-14:
                psiNew[globalID] += weight_s*f_s*exp(-k*r)/(r) 

@cuda.jit('void(float64[:,:], float64[:,:], float64[:], float64)')
def gpuHelmholtzConvolution_subtract_generic(targets,sources,psiNew,k):
    globalID = cuda.grid(1)  
    if globalID < len(targets):  
        x_t, y_t, z_t, f_t = targets[globalID][0:4]  
        psiNew[globalID] = 4*pi*f_t/k**2
        for i in range(len(sources)):  
            x_s, y_s, z_s, f_s, weight_s = sources[i]  
            r = sqrt( (x_t-x_s)**2 + (y_t-y_s)**2 + (z_t-z_s)**2 ) 
            if r > 1e-14: 
                psiNew[globalID] += weight_s*(f_s-f_t)*exp(-k*r)/(r)

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

def cpuHelmholtzSingularitySubtract(targets,sources,psiNew,k):
    for j in range(len(targets)):
        x_t, y_t, z_t, psi_t, V_t, weights_t, volume_t = targets[j]
        f_t = 2*psi_t*V_t
        for i in range(len(sources)):  # loop through all source midpoints
            x_s, y_s, z_s, psi_s, V_s, weight_s, volume_s = sources[i]  
            r = sqrt( (x_t-x_s)**2 + (y_t-y_s)**2 + (z_t-z_s)**2 )
            if r > 1e-13:
                f_s = 2*V_s*psi_s
                increment = -weight_s*(f_s-f_t)*exp(-k*r)/(4*pi*r)
#                 if abs(increment) > 1e5:
#                     print('increment = ', increment)
#                     print('target = ', targets[j])
#                     print('source = ', sources[i])
#                     print('source x = ', x_s)
#                     print('r = ',r)
                psiNew[j] += increment
        print('Target at x,y,z = ',x_t, y_t, z_t)
        print('Phi input: ', psi_t)
        print('Before analytic subtraction: psiNew[j] = ',psiNew[j])
        psiNew[j] -= f_t/k**2
        print('After analytic subtraction: psiNew[j] = ',psiNew[j])
        print()
            
    return psiNew

def cpuHelmholtzSingularitySubtract_allNumerical(targets,sources,psiNew,k):
    for j in range(len(targets)):
        x_t, y_t, z_t, psi_t, V_t, weights_t, volume_t = targets[j]
        f_t = 2*psi_t*V_t
        correction = 0.0
        for i in range(len(sources)):  # loop through all source midpoints    
            x_s, y_s, z_s, psi_s, V_s, weight_s, volume_s = sources[i]  
            r = sqrt( (x_t-x_s)**2 + (y_t-y_s)**2 + (z_t-z_s)**2 )
            if r > 1e-13:
                f_s = 2*V_s*psi_s
                increment = -weight_s*(f_s-f_t)*exp(-k*r)/(4*pi*r)
                correction += f_t*weight_s*exp(-k*r)/(4*pi*r)
#                 if abs(increment) > 1e5:
#                     print('increment = ', increment)
#                     print('target = ', targets[j])
#                     print('source = ', sources[i])
#                     print('source x = ', x_s)
#                     print('r = ',r)
                psiNew[j] += increment
        print('Target at x,y,z = ',x_t, y_t, z_t)
        print('Phi input: ', psi_t)
        print('Before analytic subtraction: psiNew[j] = ',psiNew[j])
#         psiNew[j] -= f_t/k**2
        psiNew[j] -= correction
        print('After analytic subtraction: psiNew[j] = ',psiNew[j])
        print()
            
    return psiNew

# @cuda.jit('void(float64[:,:], float64[:,:], float64[:], float64)')
# def gpuHelmholtzConvolutionSubractSingularity(targets,sources,psiNew,k):
# 
#     globalID = cuda.grid(1)  # identify the global ID of the thread
#     if globalID < len(targets):  # check that this global ID doesn't excede the number of targets
#         x_t, y_t, z_t, psi_t, V_t, weights_t, volume_t = targets[globalID]  # set the x, y, and z values of the target
#         f_t = 2*psi_t*V_t
#         psiNew[globalID] = -4*pi*f_t/k**2
# #         psiNew[globalID] = 0
# #         correction = 0.0
#         for i in range(len(sources)):  # loop through all source midpoints
#             x_s, y_s, z_s, psi_s, V_s, weight_s, volume_s = sources[i]  # set the coordinates, psi value, external potential, and volume for this source cell
#             r = sqrt( (x_t-x_s)**2 + (y_t-y_s)**2 + (z_t-z_s)**2 ) # compute the distance between target and source
# #             if (r > 1e-12 ):  # skip the convolutions when the target gridpoint = source midpoint, as G(r=r') is singular
#             if  globalID != i:  # skip the convolutions when the target gridpoint = source midpoint, as G(r=r') is singular
#                 f_s = 2*V_s*psi_s
#                 psiNew[globalID] -= weight_s*(f_s-f_t)*exp(-k*r)/r # increment the new wavefunction value
# #                 correction += weight_s*f_t*exp(-k*r)/(4*pi*r)
#         psiNew[globalID] /= 4*pi



@cuda.jit('void(float64[:,:], float64[:,:], float64[:], float64, float64)')
def gpuHelmholtzConvolutionSubractSingularity_gaussian(targets,sources,psiNew,k,a):

    globalID = cuda.grid(1)  # identify the global ID of the thread
    if globalID < len(targets):  # check that this global ID doesn't excede the number of targets
        x_t, y_t, z_t, f_t = targets[globalID][0:4]  # set the x, y, and z values of the target
        psiNew[globalID] = f_t*4*pi*(2*pi/a - (pi/a)**(3/2) * k*exp(k**2/(4*a)) * erfc( k/(2*sqrt(a)) )  )/(4*pi)
        for i in range(len(sources)):  # loop through all source midpoints
            x_s, y_s, z_s, f_s, weight_s = sources[i]  # set the coordinates, psi value, external potential, and volume for this source cell
            r = sqrt( (x_t-x_s)**2 + (y_t-y_s)**2 + (z_t-z_s)**2 ) # compute the distance between target and source
            if (r > 1e-14 ):  # skip the convolutions when the target gridpoint = source midpoint, as G(r=r') is singular
                psiNew[globalID] += weight_s*(f_s-f_t*exp(-a*r**2))*exp(-k*r)/(r) # increment the new wavefunction value
                
@cuda.jit('void(float64[:,:], float64[:,:], float64[:], float64, float64)')
def gpuHelmholtzConvolutionSubractSingularity_gaussian_no_cusp(targets,sources,psiNew,k,a):

    globalID = cuda.grid(1)  # identify the global ID of the thread
    if globalID < len(targets):  # check that this global ID doesn't excede the number of targets
        x_t, y_t, z_t, f_t = targets[globalID][0:4]  # set the x, y, and z values of the target
        psiNew[globalID] = 4*pi*f_t/(2*a)
        for i in range(len(sources)):  # loop through all source midpoints
            x_s, y_s, z_s, f_s, weight_s = sources[i]  # set the coordinates, psi value, external potential, and volume for this source cell
            r = sqrt( (x_t-x_s)**2 + (y_t-y_s)**2 + (z_t-z_s)**2 ) # compute the distance between target and source
            if (r > 1e-14 ):  # skip the convolutions when the target gridpoint = source midpoint, as G(r=r') is singular
                psiNew[globalID] += weight_s*(   f_s*exp(-k*r)-f_t*exp(-a*r**2)   )/(r) # increment the new wavefunction value
                
@cuda.jit('void(float64[:,:], float64[:,:], float64[:], float64, float64)')
def gpuHelmholtzConvolutionHybridSubractSingularity_gaussian_no_cusp(targets,sources,psiNew,k,a):

    globalID = cuda.grid(1)  # identify the global ID of the thread
    if globalID < len(targets):  # check that this global ID doesn't excede the number of targets
        x_t, y_t, z_t, f_t = targets[globalID][0:4]  # set the x, y, and z values of the target
        if x_t**2 + y_t**2 + z_t**2 < 2:
            psiNew[globalID] = 4*pi*f_t/(2*a)
        else:
            psiNew[globalID] = 0
        for i in range(len(sources)):  # loop through all source midpoints
            x_s, y_s, z_s, f_s, weight_s = sources[i]  # set the coordinates, psi value, external potential, and volume for this source cell
            r = sqrt( (x_t-x_s)**2 + (y_t-y_s)**2 + (z_t-z_s)**2 ) # compute the distance between target and source
            if (r > 1e-14 ):  # skip the convolutions when the target gridpoint = source midpoint, as G(r=r') is singular
                if x_t**2 + y_t**2 + z_t**2 < 2:
                    psiNew[globalID] += weight_s*(   f_s*exp(-k*r)-f_t*exp(-a*r**2)   )/(r) # increment the new wavefunction value
                else:
                    psiNew[globalID] += weight_s*(   f_s*exp(-k*r)   )/(r) # increment the new wavefunction value



@cuda.jit('void(float64[:,:], float64[:,:], float64[:], float64)')
def gpuHelmholtzConvolutionSubractSingularity_multDivbyr(targets,sources,psiNew,k):

    globalID = cuda.grid(1)  # identify the global ID of the thread
    if globalID < len(targets):  # check that this global ID doesn't excede the number of targets
        x_t, y_t, z_t, psi_t, V_t, weights_t, volume_t = targets[globalID]  # set the x, y, and z values of the target
        r_t = sqrt( (x_t)**2 + (y_t)**2 + (z_t)**2 ) # compute the distance between target and nucleus
        f_t = 2*psi_t*V_t
        psiNew[globalID] = -f_t/k**2
#         psiNew[globalID] = 0
#         correction = 0.0
        for i in range(len(sources)):  # loop through all source midpoints
            x_s, y_s, z_s, psi_s, V_s, weight_s, volume_s = sources[i]  # set the coordinates, psi value, external potential, and volume for this source cell
            r_s = sqrt( (x_s)**2 + (y_s)**2 + (z_s)**2 )
            r = sqrt( (x_t-x_s)**2 + (y_t-y_s)**2 + (z_t-z_s)**2 ) # compute the distance between target and source
            if (r > 1e-12 ):  # skip the convolutions when the target gridpoint = source midpoint, as G(r=r') is singular
#             if  globalID != i:  # skip the convolutions when the target gridpoint = source midpoint, as G(r=r') is singular
                f_s = 2*V_s*psi_s
                psiNew[globalID] -= weight_s*(f_s*r_t*r_s-f_t*r_t*r_s)/(r_t*r_s)*exp(-k*r)/(4*pi*r) # increment the new wavefunction value
#                 correction += weight_s*f_t*exp(-k*r)/(4*pi*r)
#         psiNew[globalID] -= correction
#         psiNew[globalID] /= r_t


@cuda.jit('void(float64[:,:], float64[:,:], float64[:], float64, float64)')
def gpuHelmholtzConvolutionSubractSingularity_k2(targets,sources,psiNew,k,k2):
    globalID = cuda.grid(1)  # identify the global ID of the thread
    if globalID < len(targets):  # check that this global ID doesn't excede the number of targets
        x_t, y_t, z_t, psi_t, V_t, weights_t, volume_t = targets[globalID]  # set the x, y, and z values of the target
        f_t = 2*psi_t*V_t
        psiNew[globalID] = -f_t/k2**2
        for i in range(len(sources)):  # loop through all source midpoints
            x_s, y_s, z_s, psi_s, V_s, weight_s, volume_s = sources[i]  # set the coordinates, psi value, external potential, and volume for this source cell
            r = sqrt( (x_t-x_s)**2 + (y_t-y_s)**2 + (z_t-z_s)**2 ) # compute the distance between target and source
            if (r > 1e-12 ):  # skip the convolutions when the target gridpoint = source midpoint, as G(r=r') is singular
                f_s = 2*V_s*psi_s
#                 psiNew[globalID] -= weight_s*(f_s-f_t)*exp(-k*r)/(4*pi*r) # increment the new wavefunction value
                psiNew[globalID] -= weight_s*( f_s*exp(-k*r)-f_t*exp(-k2*r) )/(4*pi*r) # increment the new wavefunction value

  
 
@cuda.jit('void(float64[:,:], float64[:,:], float64[:], float64)')
def gpuHelmholtzConvolutionSubractSingularity_exceptAroundNuclei(targets,sources,psiNew,k):

    globalID = cuda.grid(1)  # identify the global ID of the thread
    if globalID < len(targets):  # check that this global ID doesn't excede the number of targets
        x_t, y_t, z_t, psi_t, V_t, weights_t, volume_t = targets[globalID]  # set the x, y, and z values of the target
        r_t = (x_t**2 + y_t**2 + z_t**2)
        if r_t > 0.025:
            f_t = 2*psi_t*V_t
#         psiNew[globalID] = f_t*4*pi/k**2
            psiNew[globalID] = -f_t/k**2
        else:
            psiNew[globalID] = 0
        for i in range(len(sources)):  # loop through all source midpoints
            x_s, y_s, z_s, psi_s, V_s, weight_s, volume_s = sources[i]  # set the coordinates, psi value, external potential, and volume for this source cell
#             if not ( (x_s==x_t) and (y_s==y_t) and (z_s==z_t) ):  # skip the convolutions when the target gridpoint = source midpoint, as G(r=r') is singular
# #             if  globalID != i:  # skip the convolutions when the target gridpoint = source midpoint, as G(r=r') is singular
#                 r = sqrt( (x_t-x_s)**2 + (y_t-y_s)**2 + (z_t-z_s)**2 ) # compute the distance between target and source
#                 f_s = 2*V_s*psi_s
#                 psiNew[globalID] -= weight_s*(f_s-f_t)*exp(-k*r)/(4*pi*r) # increment the new wavefunction value
            r = sqrt( (x_t-x_s)**2 + (y_t-y_s)**2 + (z_t-z_s)**2 ) # compute the distance between target and source
#             if not ( (x_s==x_t) and (y_s==y_t) and (z_s==z_t) ):  # skip the convolutions when the target gridpoint = source midpoint, as G(r=r') is singular
            if (r > 1e-12 ):  # skip the convolutions when the target gridpoint = source midpoint, as G(r=r') is singular
#             if  globalID != i:  # skip the convolutions when the target gridpoint = source midpoint, as G(r=r') is singular
                f_s = 2*V_s*psi_s
                if r_t > 0.025:
                    psiNew[globalID] -= weight_s*(f_s-f_t)*exp(-k*r)/(4*pi*r) # increment the new wavefunction value
                else:
                    psiNew[globalID] -= weight_s*(f_s)*exp(-k*r)/(4*pi*r)

                
@cuda.jit('void(float64[:,:], float64[:,:], float64[:], float64)')
def gpuHelmholtzConvolutionSubractSingularity_Kahan(targets,sources,psiNew,k):

    globalID = cuda.grid(1)  # identify the global ID of the thread
    if globalID < len(targets):  # check that this global ID doesn't excede the number of targets
        x_t, y_t, z_t, psi_t, V_t, weights_t, volume_t = targets[globalID]  # set the x, y, and z values of the target
        f_t = 2*psi_t*V_t
#         psiNew[globalID] = f_t*4*pi/k**2
        psiNew[globalID] = -f_t/k**2
        c=0.0
        tempsum=0.0
        for i in range(len(sources)):  # loop through all source midpoints
            x_s, y_s, z_s, psi_s, V_s, weight_s, volume_s = sources[i]  # set the coordinates, psi value, external potential, and volume for this source cell
            r = sqrt( (x_t-x_s)**2 + (y_t-y_s)**2 + (z_t-z_s)**2 ) # compute the distance between target and source
            if (r > 1e-12 ):  # skip the convolutions when the target gridpoint = source midpoint, as G(r=r') is singular
                f_s = 2*V_s*psi_s
                increment = -weight_s*(f_s-f_t)*exp(-k*r)/(4*pi*r)
                y = increment-c
                t = tempsum + y
                c = (t-tempsum) - y
                tempsum = t
                
        psiNew[globalID] += tempsum # increment the new wavefunction value

@cuda.jit('void(float64[:,:], float64[:,:], float64[:])')
def gpuPoissonConvolution(targets,sources,V_hartree):
    
    globalID = cuda.grid(1)  # identify the global ID of the thread
    if globalID < len(targets):  # check that this global ID doesn't excede the number of targets
        x_t, y_t, z_t = targets[globalID][0:3] # set the x, y, and z values of the target
        V_hartree[globalID] = 0.0
        for i in range(len(sources)):  # loop through all source midpoints
            x_s, y_s, z_s, rho_s, weight_s = sources[i][0:5]  # set the coordinates, psi value, external potential, and volume for this source cell
#             if not ( abs(x_s-x_t) and (y_s==y_t) and (z_s==z_t) ):  # skip the convolutions when the target gridpoint = source midpoint, as G(r=r') is singular
            r = sqrt( (x_t-x_s)**2 + (y_t-y_s)**2 + (z_t-z_s)**2 ) # compute the distance between target and source
            if r > 1e-14:
                V_hartree[globalID] += weight_s*rho_s/r # increment the new wavefunction value
                
@cuda.jit('void(float64[:,:], float64[:,:], float64[:], float64)')
def gpuPoissonConvolutionRegularized(targets,sources,V_hartree, epsilon):
    
    globalID = cuda.grid(1)  # identify the global ID of the thread
    if globalID < len(targets):  # check that this global ID doesn't excede the number of targets
        x_t, y_t, z_t = targets[globalID][0:3] # set the x, y, and z values of the target
        V_hartree[globalID] = 0.0
        for i in range(len(sources)):  # loop through all source midpoints
            x_s, y_s, z_s, rho_s, weight_s, volume_s = sources[i]  # set the coordinates, psi value, external potential, and volume for this source cell
            r = sqrt( epsilon**2 + (x_t-x_s)**2 + (y_t-y_s)**2 + (z_t-z_s)**2 ) # compute the distance between target and source
            V_hartree[globalID] += weight_s*rho_s/r # increment the new wavefunction value
                
@cuda.jit
def gpuPoissonConvolutionSingularitySubtract(targets,sources,V_coulomb_new,k):
    
    globalID = cuda.grid(1)  # identify the global ID of the thread
    if globalID < len(targets):  # check that this global ID doesn't excede the number of targets
        x_t, y_t, z_t, rho_t = targets[globalID][0:4] # set the x, y, and z values of the target
        V_coulomb_new[globalID] = 4*pi*rho_t/k**2
        for i in range(len(sources)):  # loop through all source midpoints
            x_s, y_s, z_s, rho_s, weight_s = sources[i]  # set the coordinates, psi value, external potential, and volume for this source cell
#             if not ( abs(x_s-x_t) and (y_s==y_t) and (z_s==z_t) ):  # skip the convolutions when the target gridpoint = source midpoint, as G(r=r') is singular
            r = sqrt( (x_t-x_s)**2 + (y_t-y_s)**2 + (z_t-z_s)**2 ) # compute the distance between target and source
            if r > 1e-12:
                V_coulomb_new[globalID] += weight_s*(rho_s - rho_t*exp(-k*r) )/r # increment the new wavefunction value


               

@cuda.jit('void(float64[:,:], float64[:,:], float64[:], int64, float64, float64[:])')
def gpuPoissonConvolutionChristliebSmoothing(targets,sources,V_coulomb_new,n,epsilon, coefficients):
    globalID = cuda.grid(1)  # identify the global ID of the thread
    
                    
    if globalID < len(targets):  # check that this global ID doesn't excede the number of targets
        x_t, y_t, z_t = targets[globalID][0:3] # set the x, y, and z values of the target
        V_coulomb_new[globalID] = 0.0
        for i in range(len(sources)):  # loop through all source midpoints
            x_s, y_s, z_s, rho_s, weight_s, volume_s = sources[i]  # set the coordinates, psi value, external potential, and volume for this source cell
            scaledEpsilon = epsilon * volume_s**(1/3)
                ## COMPUTE CHRISTLIEB KERNEL APPROXIMATION ##
            r = sqrt( (x_t-x_s)**2 + (y_t-y_s)**2 + (z_t-z_s)**2 ) # compute the distance between target and source
            Gvalue = 0.0
            for ii in range(n+1):
#                 coefficient = 1.0
#                 for jj in range(ii):
#                     coefficient /= (jj+1) # this is replacing the 1/factorial(i)
#                     coefficient *= ((-1/2)-jj)
                 
                Gvalue += coefficients[ii]* (-scaledEpsilon**2)**ii * (r**2 + scaledEpsilon**2)**(-1/2-ii)
            V_coulomb_new[globalID] += weight_s*rho_s*Gvalue#/4/pi # increment the new wavefunction value

@cuda.jit('void(float64[:,:], float64[:,:], float64[:], float64, int32, float64)')
def gpuConvolutionSmoothing(targets,sources,psiNew,k,n,epsilon):
    globalID = cuda.grid(1)  # identify the global ID of the thread
    if globalID < len(targets):  # check that this global ID doesn't excede the number of targets
        x_t, y_t, z_t = targets[globalID][0:3]  # set the x, y, and z values of the target
        for i in range(len(sources)):  # loop through all source midpoints
            x_s, y_s, z_s, psi_s, V_s, weight_s, volume_s = sources[i]  # set the coordinates, psi value, external potential, and volume for this source cell
            scaledEpsilon = epsilon * volume_s**(1/3)
#             r = sqrt( (x_t-x_s)**2 + (y_t-y_s)**2 + (z_t-z_s)**2 + scaledEpsilon**2 ) # compute the distance between target and source
#             psiNew[globalID] += -2*V_s*weight_s*psi_s*exp(-k*r)/(4*pi*r)

#             scaledEpsilon = epsilon
#             if ( (skipSingular==False) or (x_t!=x_s) or (y_t!=y_s) or (z_t!=z_s) ):
#             if r > 1e-12:
                ## COMPUTE CHRISTLIEB KERNEL APPROXIMATION ##
            r = sqrt( (x_t-x_s)**2 + (y_t-y_s)**2 + (z_t-z_s)**2 ) # compute the distance between target and source
            Gvalue = 0.0
            for ii in range(n+1):
                coefficient = 1.0
                for jj in range(ii):
                    coefficient /= (jj+1) # this is replacing the 1/factorial(i)
                    coefficient *= ((-1/2)-jj)
                 
#                     Gvalue += coefficient* (-epsilon**2)**ii * (r**2 + epsilon**2)**(-1/2-ii)
                Gvalue += coefficient* (-scaledEpsilon**2)**ii * (r**2 + scaledEpsilon**2)**(-1/2-ii)
            psiNew[globalID] += -2*V_s*weight_s*psi_s*exp(-k*r)*Gvalue/4/pi # increment the new wavefunction value
            
@cuda.jit('void(float64[:,:], float64[:,:], float64[:], float64)')
def gpuHartreeIterativeSubractSingularity(targets,sources,V_Hartree_new,beta):

    globalID = cuda.grid(1)  # identify the global ID of the thread
    if globalID < len(targets):  # check that this global ID doesn't excede the number of targets
        x_t, y_t, z_t, rho_minus_betasq_Vold_t = targets[globalID][0:4] # set the x, y, and z values of the target
        V_Hartree_new[globalID] = -rho_minus_betasq_Vold_t/beta**2
        for i in range(len(sources)):  # loop through all source midpoints
            x_s, y_s, z_s, rho_minus_betasq_Vold_s, weight_s, volume_s = sources[i]  # set the coordinates, psi value, external potential, and volume for this source cell
            r = sqrt( (x_t-x_s)**2 + (y_t-y_s)**2 + (z_t-z_s)**2 ) # compute the distance between target and source
            if r > 1e-12:
                V_Hartree_new[globalID] -= weight_s * (rho_minus_betasq_Vold_s - rho_minus_betasq_Vold_t) * exp(-beta*r)/(4*pi*r) # increment the new wavefunction value
    
@cuda.jit('void(float64[:,:], float64[:,:], float64[:], float64)')
def gpuHartreeIterative(targets,sources,V_Hartree_new,beta):

    globalID = cuda.grid(1)  # identify the global ID of the thread
    if globalID < len(targets):  # check that this global ID doesn't excede the number of targets
        x_t, y_t, z_t, rho_minus_betasq_Vold_t = targets[globalID][0:4] # set the x, y, and z values of the target
        V_Hartree_new[globalID] = 0.0
        for i in range(len(sources)):  # loop through all source midpoints
            x_s, y_s, z_s, rho_minus_betasq_Vold_s, weight_s, volume_s = sources[i]  # set the coordinates, psi value, external potential, and volume for this source cell
            r = sqrt( (x_t-x_s)**2 + (y_t-y_s)**2 + (z_t-z_s)**2 ) # compute the distance between target and source
            if r > 1e-14:
                V_Hartree_new[globalID] -= weight_s*(rho_minus_betasq_Vold_s)*exp(-beta*r)/(r) # increment the new wavefunction value
                
                
@cuda.jit('void(float64[:,:], float64[:,:], float64[:], float64)')
def gpuHartreeShiftedPoisson(targets,sources,V_Hartree_new,alpha):

    globalID = cuda.grid(1)  # identify the global ID of the thread
    if globalID < len(targets):  # check that this global ID doesn't excede the number of targets
        x_t, y_t, z_t, rho_t = targets[globalID][0:4] # set the x, y, and z values of the target
        V_Hartree_new[globalID] = 0.0
        for i in range(len(sources)):  # loop through all source midpoints
            x_s, y_s, z_s, rho_s, weight_s, volume_s = sources[i]  # set the coordinates, psi value, external potential, and volume for this source cell
            r = sqrt( (x_t-x_s)**2 + (y_t-y_s)**2 + (z_t-z_s)**2 ) # compute the distance between target and source
            if r > 1e-14:
                V_Hartree_new[globalID] += weight_s*(rho_s)*exp(-alpha*r)/r # increment the new wavefunction value
                
@cuda.jit('void(float64[:,:], float64[:,:], float64[:], float64)')
def gpuHartreeShiftedPoisson_singularitySubtract(targets,sources,V_Hartree_new,alpha):
    globalID = cuda.grid(1)  # identify the global ID of the thread
    if globalID < len(targets):  # check that this global ID doesn't excede the number of targets
        x_t, y_t, z_t, rho_t = targets[globalID][0:4] # set the x, y, and z values of the target
        V_Hartree_new[globalID] = 4*pi*rho_t/alpha**2
        for i in range(len(sources)):  # loop through all source midpoints
            x_s, y_s, z_s, rho_s, weight_s, volume_s = sources[i]  # set the coordinates, psi value, external potential, and volume for this source cell
            r = sqrt( (x_t-x_s)**2 + (y_t-y_s)**2 + (z_t-z_s)**2 ) # compute the distance between target and source
            if r > 1e-14:
                V_Hartree_new[globalID] += weight_s*(rho_s-rho_t)*exp(-alpha*r)/r # increment the new wavefunction value
                
@cuda.jit('void(float64[:,:], float64[:,:], float64[:], float64)')
def gpuHartreeGaussianSingularitySubract(targets,sources,V_Hartree_new,alphasq):

    globalID = cuda.grid(1)  # identify the global ID of the thread
    if globalID < len(targets):  # check that this global ID doesn't excede the number of targets
        x_t, y_t, z_t, rho_t = targets[globalID][0:4] # set the x, y, and z values of the target
        V_Hartree_new[globalID] = rho_t* (4*pi)* alphasq/2
        for i in range(len(sources)):  # loop through all source midpoints
            x_s, y_s, z_s, rho_s, weight_s = sources[i]  # set the coordinates, psi value, external potential, and volume for this source cell
            r = sqrt( (x_t-x_s)**2 + (y_t-y_s)**2 + (z_t-z_s)**2 ) # compute the distance between target and source
            if r > 1e-14:
                V_Hartree_new[globalID] += weight_s * (rho_s -   rho_t * exp(- r*r / alphasq )   ) / r  # increment the new wavefunction value                


def dummyConvolutionToTestImportExport(targets,sources,psiNew,k):
    for i in range(len(targets)):
#         x_t, y_t, z_t = targets[i][0:3]  # set the x, y, and z values of the target
        x_s, y_s, z_s, psi_s, V_s, weight_s, volume_s = sources[i]
        psiNew[i] = psi_s
         
    return psiNew
#         for i in range(len(sources)):  # loop through all source midpoints
#             x_s, y_s, z_s, psi_s, V_s, volume_s = sources[i]  # set the coordinates, psi value, external potential, and volume for this source cell
#             if not ( (x_s==x_t) and (y_s==y_t) and (z_s==z_t) ):  # skip the convolutions when the target gridpoint = source midpoint, as G(r=r') is singular
#                 r = sqrt( (x_t-x_s)**2 + (y_t-y_s)**2 + (z_t-z_s)**2 ) # compute the distance between target and source
#                 psiNew[globalID] += -2*V_s*volume_s*psi_s*exp(-k*r)/r # increment the new wavefunction value
 