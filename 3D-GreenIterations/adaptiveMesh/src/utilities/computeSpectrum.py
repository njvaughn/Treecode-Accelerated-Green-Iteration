"""
A CUDA version of the convolution for Green Iterations.
It is written so that each thread is responsible for one of the N target gridpoints.  
Each thread interacts with all M source midpoints. 
Note: the current implementation does not use shared memory, which would provide additional
speedup.  -- 03/19/2018 NV 
"""
# from numba import cuda
# from math import sqrt,exp,factorial,pi
# import math
import numpy as np
import os
import csv
from numba import cuda, jit, njit
import time
# from convolution import gpuPoissonConvolution,gpuHelmholtzConvolutionSubractSingularity, cpuHelmholtzSingularitySubtract,cpuHelmholtzSingularitySubtract_allNumerical
import densityMixingSchemes as densityMixing
from densityMixingSchemes import AitkenAcceleration, nathanAcceleration, AitkenPointwiseAcceleration
from fermiDiracDistribution import computeOccupations
import sys
# from guppy import hpy 
import resource
sys.path.append('../ctypesTests')
sys.path.append('../ctypesTests/lib')


try:
    from convolution import *
except ImportError:
    print('Unable to import JIT GPU Convolutions')
try:
    import directSumWrappers
except ImportError:
    print('Unable to import directSumWrappers due to ImportError')
except OSError:
    print('Unable to import directSumWrappers due to OSError')
    
try:
    import treecodeWrappers
except ImportError:
    print('Unable to import treecodeWrapper due to ImportError')
except OSError:
    print('Unable to import treecodeWrapper due to OSError')
    
     
# import treecodeWrappers


@jit()
def GramMatrix(V,weights):
    n,k = np.shape(V)
    G = np.zeros((k,k))
    for i in range(k):
        for j in range(k):
            G[i,j] = np.dot(V[:,i],V[:,j]*weights)
        
    return G

def CholeskyOrthogonalize(V,weights):
    
    G = GramMatrix(V, weights)
    print('Gram Matrix: ', G)
    
    L = np.linalg.cholesky(G)
    print('Cholesky L: ', L)
    
    orthV = np.dot( V, L.T)
    
    return orthV

@jit(parallel=True)
def modifiedGramSchmidt(V,weights):
    n,k = np.shape(V)
    U = np.zeros_like(V)
    U[:,0] = V[:,0] / np.dot(V[:,0],V[:,0]*weights)
    for i in range(1,k):
        U[:,i] = V[:,i]
        for j in range(i):
#             print('Orthogonalizing %i against %i' %(i,j))
            U[:,i] -= (np.dot(U[:,i],U[:,j]*weights) / np.dot(U[:,j],U[:,j]*weights))*U[:,j]
        U[:,i] /= np.dot(U[:,i],U[:,i]*weights)
        
    return U

@jit()
def modifiedGramSchmidt_singleOrbital(V,weights,targetOrbital, n, k):
    U = V[:,targetOrbital]
    for j in range(targetOrbital):
#         print('Orthogonalizing %i against %i' %(targetOrbital,j))
#         U -= (np.dot(V[:,targetOrbital],V[:,j]*weights) / np.dot(V[:,j],V[:,j]*weights))*V[:,j]
        U -= np.dot(V[:,targetOrbital],V[:,j]*weights) *V[:,j]
        U /= np.sqrt( np.dot(U,U*weights) )
    
    U /= np.sqrt( np.dot(U,U*weights) )  # normalize again at end (safegaurd for the zeroth orbital, which doesn't enter the above loop)
        
    return U

def modifiedGramSchrmidt_noNormalization(V,weights):
    n,k = np.shape(V)
    U = np.zeros_like(V)
    U[:,0] = V[:,0] 
    for i in range(1,k):
        U[:,i] = V[:,i]
        for j in range(i):
            print('Orthogonalizing %i against %i' %(i,j))
            U[:,i] -= (np.dot(U[:,i],U[:,j]*weights) / np.dot(U[:,j],U[:,j]*weights))*U[:,j]
#         U[:,i] /= np.dot(U[:,i],U[:,i]*weights)
        
    return U

def normalizeOrbitals(V,weights):
    print('Only normalizing, not orthogonalizing orbitals')
    n,k = np.shape(V)
    U = np.zeros_like(V)
#     U[:,0] = V[:,0] / np.dot(V[:,0],V[:,0]*weights)
    for i in range(0,k):
        U[:,i]  = V[:,i]
        U[:,i] /= np.sqrt( np.dot(U[:,i],U[:,i]*weights) )
        
        if abs( 1- np.dot(U[:,i],U[:,i]*weights)) > 1e-12:
            print('orbital ', i, ' not normalized? Should be 1: ', np.dot(U[:,i],U[:,i]*weights))
    
    return U

# def wavefunctionErrors(wave1, wave2, weights, x,y,z):
#     L2error = np.sqrt( np.sum(weights*(wave1-wave2)**2 ) )
#     LinfIndex = np.argmax( abs( wave1-wave2 ))
#     Linf = wave1[LinfIndex] - wave2[LinfIndex] 
#     LinfRel = (wave1[LinfIndex] - wave2[LinfIndex])/wave1[LinfIndex]
#     xinf = x[LinfIndex]
#     yinf = y[LinfIndex]
#     zinf = z[LinfIndex]
#     
#     print('~~~~~~~~Wavefunction Errors~~~~~~~~~~')
#     print("L2 Error:             ", L2error)
#     print("Linf Error:           ", Linf)
#     print("LinfRel Error:        ", LinfRel)
#     print("Located at:           ", xinf, yinf, zinf)
#     print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

xi=yi=zi=-1.1
xf=yf=zf=1.1
numpts=3000

def computeSpectrum(tree, nOrbitals, targetEnergyEigenvalue, intraScfTolerance, interScfTolerance, numberOfTargets, gradientFree, symmetricIteration, GPUpresent, 
                    treecode, treecodeOrder, theta, maxParNode, batchSize,
                    mixingScheme, mixingParameter, mixingHistoryCutoff,
                    subtractSingularity, gaussianAlpha, inputFile='',outputFile='',restartFile=False,
                    onTheFlyRefinement = False, vtkExport=False, outputErrors=False, maxOrbitals=None, maxSCFIterations=None): 
    '''
    Green Iterations for Kohn-Sham DFT using Clenshaw-Curtis quadrature.
    '''
    
#     return
    print('MEMORY USAGE: ', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    print()
    
    tree.nOrbitals = nOrbitals
    tree.orbitalEnergies = np.ones(nOrbitals)

    if hasattr(tree, 'referenceEigenvalues'):
        referenceEigenvalues = tree.referenceEigenvalues
    else:
        print('Tree did not have attribute referenceEigenvalues')
        referenceEigenvalues = np.zeros(tree.nOrbitals)
        return
    
    # Store Tree variables locally
    numberOfGridpoints = tree.numberOfGridpoints
    gaugeShift = tree.gaugeShift
    Temperature = 200  # set to 200 Kelvin
    
    

    greenIterationOutFile = outputFile[:-4]+'_GREEN_'+str(tree.numberOfGridpoints)+outputFile[-4:]
    SCFiterationOutFile =   outputFile[:-4]+'_SCF_'+str(tree.numberOfGridpoints)+outputFile[-4:]
    densityPlotsDir =       outputFile[:-4]+'_SCF_'+str(tree.numberOfGridpoints)+'_plots'



    Eold = -10
    SCFcount=0
    tree.totalIterationCount = 0
    
    # Initialize orbital matrix
    targets = tree.extractLeavesDensity()
    orbitals = np.random.rand(len(targets),tree.nOrbitals)
    oldOrbitals = np.random.rand(len(targets),tree.nOrbitals)
    
          
    for m in range(nOrbitals):
        targets = tree.extractPhi(m)
        weights = np.copy(targets[:,5])
        tree.importPhiOnLeaves(orbitals[:,m],m)

    targets = tree.extractLeavesDensity() 
    weights = targets[:,4]

    targets = tree.extractLeavesDensity() 
    weights = targets[:,4]
    
        
    

    threadsPerBlock = 512
    blocksPerGrid = (numberOfTargets + (threadsPerBlock - 1)) // threadsPerBlock  # compute the number of blocks based on N and threadsPerBlock
    
    print('\nEntering computeSpectrum()')
    print('\nNumber of targets:   ', numberOfTargets)
    print('Threads per block:   ', threadsPerBlock)
    print('Blocks per grid:     ', blocksPerGrid)
    
    densityResidual = 10                                   # initialize the densityResidual to something that fails the convergence tolerance

#     [Etrue, ExTrue, EcTrue, Eband] = np.genfromtxt(inputFile,dtype=[(str,str,int,int,float,float,float,float,float)])[4:8]
    [Eband, Ekinetic, Eexchange, Ecorrelation, Ehartree, Etotal] = np.genfromtxt(inputFile)[3:9]
    print([Eband, Ekinetic, Eexchange, Ecorrelation, Ehartree, Etotal])

    ### COMPUTE THE INITIAL HAMILTONIAN ###
    density_targets = tree.extractLeavesDensity()  
    density_sources = np.copy(density_targets)
#     sources = tree.extractDenstiySecondaryMesh()   # extract density on secondary mesh

    integratedDensity = np.sum( density_sources[:,3]*density_sources[:,4] )
#     densityResidual = np.sqrt( np.sum( (sources[:,3]-oldDensity[:,3])**2*weights ) )
    print('Integrated density: ', integratedDensity)

#     starthartreeConvolutionTime = timer()
    alpha = gaussianAlpha
    alphasq=alpha*alpha
    
    
#     print('Using Gaussian singularity subtraction, alpha = ', alpha)
    
    print('GPUpresent set to ', GPUpresent)
    print('Type: ', type(GPUpresent))
    if GPUpresent==False:
        numTargets = len(density_targets)
        numSources = len(density_sources)
#         print('numTargets = ', numTargets)
#         print(targets[:10,:])
#         print('numSources = ', numSources)
#         print(sources[:10,:])
        copystart = time.time()
        sourceX = np.copy(density_sources[:,0])
#         print(np.shape(sourceX))
#         print('sourceX = ', sourceX[0:10])
        sourceY = np.copy(density_sources[:,1])
        sourceZ = np.copy(density_sources[:,2])
        sourceValue = np.copy(density_sources[:,3])
        sourceWeight = np.copy(density_sources[:,4])
        
        targetX = np.copy(density_targets[:,0])
        targetY = np.copy(density_targets[:,1])
        targetZ = np.copy(density_targets[:,2])
        targetValue = np.copy(density_targets[:,3])
        targetWeight = np.copy(density_targets[:,4])
        copytime=time.time()-copystart
        print('Copy time before convolution: ', copytime)
        start = time.time()
        
        if treecode==False:
            V_hartreeNew = directSumWrappers.callCompiledC_directSum_PoissonSingularitySubtract(numTargets, numSources, alphasq, 
                                                                                                  targetX, targetY, targetZ, targetValue,targetWeight, 
                                                                                                  sourceX, sourceY, sourceZ, sourceValue, sourceWeight)

            V_hartreeNew += targets[:,3]* (4*np.pi)/ alphasq/ 2   # Correct for exp(-r*r/alphasq)  # DONT TRUST

        elif treecode==True:
            
            
# #         V_hartreeNew += targets[:,3]* (4*np.pi)* alphasq/2  # Wrong


#         V_hartreeNew = directSumWrappers.callCompiledC_directSum_Poisson(numTargets, numSources, 
#                                                                         targetX, targetY, targetZ, targetValue,targetWeight, 
#                                                                         sourceX, sourceY, sourceZ, sourceValue, sourceWeight)

            potentialType=2 # shoud be 2 for Hartree w/ singularity subtraction.  Set to 0, 1, or 3 just to test other kernels quickly
            alpha = gaussianAlpha
            V_hartreeNew = treecodeWrappers.callTreedriver(numTargets, numSources, 
                                                           targetX, targetY, targetZ, targetValue, 
                                                           sourceX, sourceY, sourceZ, sourceValue, sourceWeight,
                                                           potentialType, alpha, treecodeOrder, theta, maxParNode, batchSize)
               
            if potentialType==2:
                V_hartreeNew += targets[:,3]* (4*np.pi) / alphasq/2

        
#         print('First few terms of V_hartreeNew: ', V_hartreeNew[:8])
#         print('Convolution time: ', time.time()-start)
        
        
        
        
    elif GPUpresent==True:
        if treecode==False:
            V_hartreeNew = np.zeros((len(density_targets)))
            start = time.time()
            gpuHartreeGaussianSingularitySubract[blocksPerGrid, threadsPerBlock](density_targets,density_sources,V_hartreeNew,alphasq)
#             print('Convolution time: ', time.time()-start)
#             return
        elif treecode==True:
            copystart=time.time()
            numTargets = len(density_targets)
            numSources = len(density_sources)
            sourceX = np.copy(density_sources[:,0])

            sourceY = np.copy(density_sources[:,1])
            sourceZ = np.copy(density_sources[:,2])
            sourceValue = np.copy(density_sources[:,3])
            sourceWeight = np.copy(density_sources[:,4])
            
            targetX = np.copy(density_targets[:,0])
            targetY = np.copy(density_targets[:,1])
            targetZ = np.copy(density_targets[:,2])
            targetValue = np.copy(density_targets[:,3])
            targetWeight = np.copy(density_targets[:,4])
            copytime = time.time()-copystart
            print('Copy time before calling treecode: ', copytime)
            start = time.time()
            potentialType=2 
            alpha = gaussianAlpha
            V_hartreeNew = treecodeWrappers.callTreedriver(numTargets, numSources, 
                                                           targetX, targetY, targetZ, targetValue, 
                                                           sourceX, sourceY, sourceZ, sourceValue, sourceWeight,
                                                           potentialType, alpha, treecodeOrder, theta, maxParNode, batchSize)
#             print('Convolution time: ', time.time()-start)
            
        else:
            print('treecode True or False?')
            return
    

#     hartreeConvolutionTime = timer() - starthartreeConvolutionTime
#     print('Computing Vhartree took:    %.4f seconds. ' %hartreeConvolutionTime)
    tree.importVhartreeOnLeaves(V_hartreeNew)
    tree.updateVxcAndVeffAtQuadpoints()
    
    

    for m in range(nOrbitals):
        # fill in orbitals
        targets = tree.extractPhi(m)
        weights = np.copy(targets[:,5])
        oldOrbitals[:,m] = np.copy(targets[:,3])
        orbitals[:,m] = np.copy(targets[:,3])


    initialWaveData = tree.extractPhi(0)
    initialPsi0 = np.copy(initialWaveData[:,3])
    x = np.copy(initialWaveData[:,0])
    y = np.copy(initialWaveData[:,1])
    z = np.copy(initialWaveData[:,2])
    


    inputIntraSCFtolerance = np.copy(intraScfTolerance)

    energyResidual=1
    
    residuals = 10*np.ones_like(tree.orbitalEnergies)

    
    while max(residuals) > intraScfTolerance:
    
    
        for m in range(nOrbitals): 
            
                
            orbitalResidual = 1
            greenIterationsCount = 1
            max_GreenIterationsCount = 15000  # set very high.  Don't ever stop green iterations before convergence.
                
    
            
#             print('Working on orbital %i' %m)
            inputIntraSCFtolerance = np.copy(intraScfTolerance)
            orbitalResidual = 1.0
            oldOrbitalResidual = 2.0
            oldOldOrbitalResidual = 4.0
            psiNewNorm = 10
            previousResidual = 1
            previousEigenvalueDiff = 1
            
            normDiff=1.0
            eigenvalueDiff=1.0
            
            
            ratioTol = 5e-3000
            
            previousResidualRatio = 2
            previousEigenvalueResidualRatio = 2
            
        
            tree.totalIterationCount += 1
#                     orbitalResidual = 0.0
            
            oldOldOrbitalResidual = oldOrbitalResidual
            oldOrbitalResidual = orbitalResidual
            

            sources = tree.extractPhi(m)
            targets = np.copy(sources)
            weights = np.copy(targets[:,5])
                
                
                        
                        
            oldOrbitals[:,m] = np.copy(targets[:,3])


            sources = tree.extractGreenIterationIntegrand(m)
#                 sources = tree.extractGreenIterationIntegrand_Deflated(m,orbitals,weights)
            
            
            targets = np.copy(sources)



            oldEigenvalue =  tree.orbitalEnergies[m] 

            k = np.sqrt(-2*targetEnergyEigenvalue)  # same k for all eigenfunctions.
            
            phiNew = np.zeros((len(targets)))
            if subtractSingularity==0: 
#                 print('Using singularity skipping')
                gpuHelmholtzConvolution[blocksPerGrid, threadsPerBlock](targets,sources,phiNew,k) 
            elif subtractSingularity==1:
                    
                    
                    
                if treecode==False:
                    startTime = time.time()
                    if symmetricIteration==False:
                        gpuHelmholtzConvolutionSubractSingularity[blocksPerGrid, threadsPerBlock](targets,sources,phiNew,k) 
                        convolutionTime = time.time()-startTime
#                         phiNew /= (4*np.pi)
#                         print('Using asymmetric singularity subtraction.  Convolution time: ', convolutionTime)
                    elif symmetricIteration==True:
                        gpuHelmholtzConvolutionSubractSingularitySymmetric[blocksPerGrid, threadsPerBlock](targets,sources,sqrtV,phiNew,k) 
                        phiNew *= -1
                        convolutionTime = time.time()-startTime
#                         print('Using symmetric singularity subtraction.  Convolution time: ', convolutionTime)

                    
                elif treecode==True:
                    
                    copyStart = time.time()
                    numTargets = len(targets)
                    numSources = len(sources)

                    sourceX = np.copy(sources[:,0])

                    sourceY = np.copy(sources[:,1])
                    sourceZ = np.copy(sources[:,2])
                    sourceValue = np.copy(sources[:,3])
                    sourceWeight = np.copy(sources[:,4])
                    
                    targetX = np.copy(targets[:,0])
                    targetY = np.copy(targets[:,1])
                    targetZ = np.copy(targets[:,2])
                    targetValue = np.copy(targets[:,3])
                    targetWeight = np.copy(targets[:,4])
                
                    copytime=time.time()-copyStart
#                                         print('Time spent copying arrays for treecode call: ', copytime)
                    potentialType=3
                    kappa = k
                    start = time.time()
                    phiNew = treecodeWrappers.callTreedriver(numTargets, numSources, 
                                                                   targetX, targetY, targetZ, targetValue, 
                                                                   sourceX, sourceY, sourceZ, sourceValue, sourceWeight,
                                                                   potentialType, kappa, treecodeOrder, theta, maxParNode, batchSize)
#                     print('Convolution time: ', time.time()-start)
                    phiNew /= (4*np.pi)
                
                else: 
                    print('treecode true or false?')
                    return
                
            else:
                print('Invalid option for singularitySubtraction, should be 0 or 1.')
                return
                    
                    


            # Compute Rayleigh Quotients
            orbitals[:,m] = np.copy(phiNew)
#             print('Norm of psiNew: ', np.sqrt( np.sum(phiNew*phiNew*weights) ) )
#             tree.orbitalEnergies[m] = np.sum( phiNew*phiNew*weights ) / np.sum( oldOrbitals[:,m]*oldOrbitals[:,m]*weights )
           
                   
        for m in range(tree.nOrbitals):
            # Compute Rayleigh Quotients
            tree.orbitalEnergies[m] = np.sign( np.sum( oldOrbitals[:,m]*orbitals[:,m]*weights ) ) * np.sqrt( abs( np.sum( oldOrbitals[:,m]*orbitals[:,m]*weights ) ) )  / np.sqrt( np.sum( oldOrbitals[:,m]*oldOrbitals[:,m]*weights ) )
                         
            # Orthogonalize
            orthWavefunction = modifiedGramSchmidt_singleOrbital(orbitals,weights,m, tree.numberOfGridpoints, nOrbitals)
            orbitals[:,m] = np.copy(orthWavefunction) 
            tree.importPhiOnLeaves(orbitals[:,m], m)
            
            # Compute residuals
#             residuals[m] = np.sqrt( np.sum( (orbitals[:,m]-oldOrbitals[:,m])**2*weights ) )
            residuals[m] = min( np.sqrt( np.sum( (orbitals[:,m]-oldOrbitals[:,m])**2*weights ) ), np.sqrt( np.sum( (orbitals[:,m]+oldOrbitals[:,m])**2*weights ) ) )
            
            

            
        
        print('residuals = ')
        print(residuals)
        print('eigenvalues = ')
        print(tree.orbitalEnergies)
        print()
                

        header = ['targetOrbital', 'Iteration', 'orbitalResiduals', 'energyEigenvalues', 'eigenvalueResidual']

        myData = [m, greenIterationsCount, residuals,
                  tree.orbitalEnergies-tree.gaugeShift, eigenvalueDiff]

        if not os.path.isfile(greenIterationOutFile):
            myFile = open(greenIterationOutFile, 'a')
            with myFile:
                writer = csv.writer(myFile)
                writer.writerow(header) 
            
        
        myFile = open(greenIterationOutFile, 'a')
        with myFile:
            writer = csv.writer(myFile)
            writer.writerow(myData)



#             tree.sortOrbitalsAndEnergies() 
        greenIterationsCount += 1 
                
            
      



