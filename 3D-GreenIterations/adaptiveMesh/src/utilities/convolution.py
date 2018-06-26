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

from hydrogenPotential import trueEnergy, trueWavefunction
    
# @cuda.jit
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
def gpuConvolutionSubractSingularity(targets,sources,psiNew,k):
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
                        
            
def greenIterations_CC(tree, energyLevel, residualTolerance, numberOfTargets, subtractSingularity, smoothingN, smoothingEps, normalizationFactor=1, threadsPerBlock=512, visualize=False, outputErrors=False):  # @DontTrace
    '''
    :param residualTolerance: exit condition for Green Iterations, residual on the total energy
    :param energyLevel: energy level trying to compute
    '''
    blocksPerGrid = (numberOfTargets + (threadsPerBlock - 1)) // threadsPerBlock  # compute the number of blocks based on N and threadsPerBlock
    print('\nEntering greenIterations_CC()')
    print('\nNumber of targets:   ', numberOfTargets)
    print('Threads per block:   ', threadsPerBlock)
    print('Blocks per grid:     ', blocksPerGrid)
    
    if visualize == True:
        currentDirectory = os.getcwd()
        try: shutil.rmtree(currentDirectory+'/plots')
        except OSError:
            print(OSError)
        os.mkdir(currentDirectory+'/plots')
        tree.wavefunctionSlice(0.0,n=energyLevel,scalingFactor=normalizationFactor,saveID = currentDirectory+'/plots/%04i'%0)
    
    GIcounter=1                                     # initialize the counter to counter the number of iterations required for convergence
    residual = 1                                    # initialize the residual to something that fails the convergence tolerance
    Eold = -10.0
    
    Etrue = trueEnergy(energyLevel)
    
    while ( (residual > residualTolerance) and (GIcounter<50) ):
#     while GIcounter < 15:
        print()
        print('Green Iteration Count ', GIcounter)
        GIcounter+=1
        startExtractionTime = timer()
        sources = tree.extractLeavesAllGridpoints()  # extract the source point locations.  Currently, these are just all the leaf midpoints
        targets = tree.extractLeavesAllGridpoints()  # extract the target point locations.  Currently, these are all 27 gridpoints per cell (no redundancy)
#         targets = sources
#         self.assertEqual(tree.numberOfGridpoints, len(targets), "targets not equal to number of gridpoints") # verify that the number of targets equals the number of total gridpoints of the tree
#         maxdiff = np.max(np.abs(sources[:][0] + 1.77635684e-15))
#         print('maxdiff = ', maxdiff)
        ExtractionTime = timer() - startExtractionTime
        psiNew = np.zeros((len(targets)))
        startConvolutionTime = timer()
        k = np.sqrt(-2*tree.E)                 # set the k value coming from the current guess for the energy

        if subtractSingularity==0:
            gpuConvolution[blocksPerGrid, threadsPerBlock](targets,sources,psiNew,k)  # call the GPU convolution
        elif subtractSingularity==1:
            gpuConvolutionSubractSingularity[blocksPerGrid, threadsPerBlock](targets,sources,psiNew,k)  # call the GPU convolution
#         psiNew = cpuConvolution(targets, sources, psiNew, k)

#         skipSingular=True  # set this to TRUE to reproduce results when I was skipping singularity.
#         gpuConvolutionSmoothing[blocksPerGrid, threadsPerBlock](targets,sources,psiNew,k,smoothingN,smoothingEps,skipSingular)
        ConvolutionTime = timer() - startConvolutionTime
        print('Extraction took:             %.4f seconds. ' %ExtractionTime)
        print('Convolution took:            %.4f seconds. ' %ConvolutionTime)


        tree.importPsiOnLeaves(psiNew)         # import the new wavefunction values into the tree.
#         tree.normalizeWavefunction()
#         for i in range(energyLevel):
#             tree.orthogonalizeWavefunction(i)
        tree.normalizeWavefunction()           # Normalize the new wavefunction 
        tree.computeWaveErrors(energyLevel,normalizationFactor)    # Compute the wavefunction errors compared to the analytic ground state 
        print('Convolution wavefunction errors: %.10e L2,  %.10e max' %(tree.L2NormError, tree.maxPointwiseError))
#         tempSources = tree.extractLeavesMidpointsOnly()
#         print('Printing a few wavefunction values...\n','x, y, z, psi\n', tempSources[2000:2005][:3])
        startEnergyTime = timer()
        tree.updateEnergy()  
#         tree.E = -0.5 # force E to the analytic value in order to test wavefunction accuracy                
        energyUpdateTime = timer() - startEnergyTime
        print('Energy Update took:              %.4f seconds. ' %energyUpdateTime)
        residual = abs(Eold - tree.E)  # Compute the residual for determining convergence
        print('Energy Residual:                 %.3e' %residual)

        Eold = tree.E
        print('Updated Potential Value:         %.10f Hartree, %.10e error' %(tree.totalPotential, -1.0 - tree.totalPotential))
        print('Updated Kinetic Value:           %.10f Hartree, %.10e error' %(tree.totalKinetic, 0.5 - tree.totalKinetic))
        print('Updated Energy Value:            %.10f Hartree, %.10e error' %(tree.E, tree.E-Etrue))
        if tree.E > 0.0:                       # Check that the current guess for energy didn't go positive.  Reset it if it did. 
            print('Warning, Energy is positive')
            tree.E = -0.5
        if visualize == True:
#             tree.visualizeMesh('psi')
            tree.wavefunctionSlice(0.0,n=energyLevel,scalingFactor=normalizationFactor,saveID = currentDirectory+'/plots/%04i'%(GIcounter-1))
    print('\nConvergence to a tolerance of %f took %i iterations' %(residualTolerance, GIcounter))
    
#     if outputErrors == True:
#         energyError  = tree.E, tree.E-Etrue
#         psiL2Error   = tree.L2NormError
#         psiLinfError = tree.maxPointwiseError
#         return energyError, psiL2, psiLinf


def greenIterations(tree, energyLevel, residualTolerance, numberOfTargets, subtractSingularity, smoothingN, smoothingEps, normalizationFactor=1, threadsPerBlock=512, visualize=False, outputErrors=False):  # @DontTrace
    '''
    :param residualTolerance: exit condition for Green Iterations, residual on the total energy
    :param energyLevel: energy level trying to compute
    '''
    blocksPerGrid = (numberOfTargets + (threadsPerBlock - 1)) // threadsPerBlock  # compute the number of blocks based on N and threadsPerBlock
    print('\nNumber of targets:   ', numberOfTargets)
    print('Threads per block:   ', threadsPerBlock)
    print('Blocks per grid:     ', blocksPerGrid)
    
    if visualize == True:
        currentDirectory = os.getcwd()
        try: shutil.rmtree(currentDirectory+'/plots')
        except OSError:
            print(OSError)
        os.mkdir(currentDirectory+'/plots')
        tree.wavefunctionSlice(0.0,n=energyLevel,scalingFactor=normalizationFactor,saveID = currentDirectory+'/plots/%04i'%0)
    
    GIcounter=1                                     # initialize the counter to counter the number of iterations required for convergence
    residual = 1                                    # initialize the residual to something that fails the convergence tolerance
    Eold = -10.0
    
    Etrue = trueEnergy(energyLevel)
    
    while ( (residual > residualTolerance) and (GIcounter<50) ):
#     while GIcounter < 15:
        print()
        print('Green Iteration Count ', GIcounter)
        GIcounter+=1
        startExtractionTime = timer()
        sources = tree.extractLeavesMidpointsOnly()  # extract the source point locations.  Currently, these are just all the leaf midpoints
        targets = tree.extractLeavesAllGridpoints()  # extract the target point locations.  Currently, these are all 27 gridpoints per cell (no redundancy)
#         self.assertEqual(tree.numberOfGridpoints, len(targets), "targets not equal to number of gridpoints") # verify that the number of targets equals the number of total gridpoints of the tree
        ExtractionTime = timer() - startExtractionTime
        psiNew = np.zeros((len(targets)))
        startConvolutionTime = timer()
        k = np.sqrt(-2*tree.E)                 # set the k value coming from the current guess for the energy
#             k = 1
#         if subtractSingularity==0:
#             gpuConvolution[blocksPerGrid, threadsPerBlock](targets,sources,psiNew,k)  # call the GPU convolution
#         elif subtractSingularity==1:
#             gpuConvolutionSubractSingularity[blocksPerGrid, threadsPerBlock](targets,sources,psiNew,k)  # call the GPU convolution


        skipSingular=False  # set this to TRUE to reproduce results when I was skipping singularity.
        gpuConvolutionSmoothing[blocksPerGrid, threadsPerBlock](targets,sources,psiNew,k,smoothingN,smoothingEps,skipSingular)
        ConvolutionTime = timer() - startConvolutionTime
        print('Extraction took:             %.4f seconds. ' %ExtractionTime)
        print('Convolution took:            %.4f seconds. ' %ConvolutionTime)


        tree.importPsiOnLeaves(psiNew)         # import the new wavefunction values into the tree.
#         tree.normalizeWavefunction()
        for i in range(energyLevel):
            tree.orthogonalizeWavefunction(i)
        tree.normalizeWavefunction()           # Normalize the new wavefunction 
        tree.computeWaveErrors(energyLevel,normalizationFactor)    # Compute the wavefunction errors compared to the analytic ground state 
        print('Convolution wavefunction errors: %.10e L2,  %.10e max' %(tree.L2NormError, tree.maxPointwiseError))
#         tempSources = tree.extractLeavesMidpointsOnly()
#         print('Printing a few wavefunction values...\n','x, y, z, psi\n', tempSources[2000:2005][:3])
        startEnergyTime = timer()
        tree.updateEnergy()  
#         tree.E = -0.5 # force E to the analytic value in order to test wavefunction accuracy                
        energyUpdateTime = timer() - startEnergyTime
        print('Energy Update took:              %.4f seconds. ' %energyUpdateTime)
        residual = abs(Eold - tree.E)  # Compute the residual for determining convergence
        print('Energy Residual:                 %.3e' %residual)

        Eold = tree.E
        print('Updated Potential Value:         %.10f Hartree, %.10e error' %(tree.totalPotential, -1.0 - tree.totalPotential))
        print('Updated Kinetic Value:           %.10f Hartree, %.10e error' %(tree.totalKinetic, 0.5 - tree.totalKinetic))
        print('Updated Energy Value:            %.10f Hartree, %.10e error' %(tree.E, tree.E-Etrue))
        if tree.E > 0.0:                       # Check that the current guess for energy didn't go positive.  Reset it if it did. 
            print('Warning, Energy is positive')
            tree.E = -1.0
        if visualize == True:
#             tree.visualizeMesh('psi')
            tree.wavefunctionSlice(0.0,n=energyLevel,scalingFactor=normalizationFactor,saveID = currentDirectory+'/plots/%04i'%(GIcounter-1))
    print('\nConvergence to a tolerance of %f took %i iterations' %(residualTolerance, GIcounter))
    