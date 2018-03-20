"""
A CUDA version of the convolution for Green Iterations.
It is written so that each thread is responsible for one of the N target gridpoints.  
Each thread interacts with all M source midpoints. 
Note: the current implementation does not use shared memory, which would provide additional
speedup.  -- 03/19/2018 NV 
"""
from numba import cuda
from math import sqrt,exp
        
@cuda.jit
def gpuConvolution(targets,sources,psiNew,k):
    globalID = cuda.grid(1)  # identify the global ID of the thread
    if globalID < len(targets):  # check that this global ID doesn't excede the number of targets
        x_t, y_t, z_t = targets[globalID][0:3]  # set the x, y, and z values of the target
        for i in range(len(sources)):  # loop through all source midpoints
            x_s, y_s, z_s, psi_s, V_s, volume_s = sources[i]  # set the coordinates, psi value, external potential, and volume for this source cell
            if not ( (x_s==x_t) and (y_s==y_t) and (z_s==z_t) ):  # skip the convolutions when the target gridpoint = source midpoint, as G(r=r') is singular
                r = sqrt( (x_t-x_s)**2 + (y_t-y_s)**2 + (z_t-z_s)**2 ) # compute the distance between target and source
                psiNew[globalID] += -2*V_s*volume_s*psi_s*exp(-k*r)/r # increment the new wavefunction value
            