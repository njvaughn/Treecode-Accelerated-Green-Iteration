'''
Routines for performing eigenvalue-one algorithm, which was used for spectral analysis of the integral operator.

'''
import inspect
import numpy as np
import time
import csv
import resource
import GPUtil
import os


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
    

# try:
#     from orthogonalizationRoutines import *
# except ImportError:
from orthogonalizationRoutines import modifiedGramSchmidt_singleOrbital as mgs


def eigenvalueOne_FixedPoint_Closure(gi_args):
#     gi_args_out = {}
    def eigenvalueOne_FixedPoint(psiIn, gi_args):
        # what other things do we need?  Energies, Times, orbitals, Veff, runtime constants (symmetricIteration, GPUpresent, subtractSingularity, treecode, outputfiles, ...)  
        
        
        ## UNPACK GIARGS
#         print('MEMORY USAGE: ', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss )
#         GPUtil.showUtilization()
        orbitals = gi_args['orbitals']
        oldOrbitals = gi_args['oldOrbitals']
        Energies = gi_args['Energies']
        Times = gi_args['Times']
        Veff = gi_args['Veff']
        m = gi_args['m']
        symmetricIteration = gi_args['symmetricIteration']
        GPUpresent = gi_args['GPUpresent']
        subtractSingularity = gi_args['subtractSingularity']
        treecode = gi_args['treecode']
        treecodeOrder = gi_args['treecodeOrder']
        theta = gi_args['theta']
        maxParNode=gi_args['maxParNode']
        batchSize=gi_args['batchSize']
        nPoints = gi_args['nPoints']
        X = gi_args['X']
        Y = gi_args['Y']
        Z = gi_args['Z']  
        W = gi_args['W']
        gradientFree = gi_args['gradientFree']
        SCFcount = gi_args['SCFcount']
        greenIterationsCount = gi_args['greenIterationsCount']
        residuals = gi_args['residuals']
        greenIterationOutFile = gi_args['greenIterationOutFile']
        threadsPerBlock=gi_args['threadsPerBlock']
        blocksPerGrid=gi_args['blocksPerGrid']
        referenceEigenvalues = gi_args['referenceEigenvalues']

        
        print('Who called F(x)? ', inspect.stack()[2][3])
        inputWave = np.copy(psiIn[:-1])
    
        # global data structures
#         global orbitals, oldOrbitals, residuals, eigenvalueHistory, Veff, Energies, referenceEigenvalues
    #     global X, Y, Z, W
        
        # Global constants and counters
#         global threadsPerBlock, blocksPerGrid, SCFcount, greenIterationsCount
#         global greenIterationOutFile 
        
        Times['totalIterationCount'] += 1
        
        
         
        oldOrbitals[:,m] = np.copy(psiIn[:-1])    
        orbitals[:,m] = np.copy(psiIn[:-1])
#         n,M = np.shape(orbitals)
#         orthWavefunction = modifiedGramSchmidt_singleOrbital(orbitals,W,m, n, M)
#         orbitals[:,m] = np.copy(orthWavefunction)
        Energies['orbitalEnergies'][m] = np.copy(psiIn[-1])
        
    
     
        f = -2*orbitals[:,m]*Veff

        
        
        oldEigenvalue =  Energies['orbitalEnergies'][m] 
#         k = np.sqrt(-2*Energies['orbitalEnergies'][m])

        if gi_args['targetEpsilon']<(gi_args['referenceEigenvalues'][m]+Energies['gaugeShift']):
            k = np.sqrt(-2*gi_args['targetEpsilon'])
        else:
            k = np.sqrt(-2*(gi_args['referenceEigenvalues'][m]+Energies['gaugeShift']))
        
        print('k = ', k)
    
        phiNew = np.zeros(nPoints)
        if subtractSingularity==0: 
            print('Using singularity skipping')
            gpuHelmholtzConvolution[blocksPerGrid, threadsPerBlock](targets,sources,phiNew,k) 
        elif subtractSingularity==1:
            if Energies['orbitalEnergies'][m] < 10.25**100: 
                
                
                if GPUpresent==False:
                    startTime=time.time()
                    potentialType=3
                    kappa = k
                    startTime = time.time()
                    numDevices=0
                    numThreads=4
                    phiNew = treecodeWrappers.callTreedriver(nPoints, nPoints, 
                                                                   np.copy(X), np.copy(Y), np.copy(Z), np.copy(f), 
                                                                   np.copy(X), np.copy(Y), np.copy(Z), np.copy(f), np.copy(W),
                                                                   potentialType, kappa, treecodeOrder, theta, maxParNode, batchSize, numDevices, numThreads)
                    phiNew /= (4*np.pi)
                    convolutionTime = time.time()-startTime
                    print('Using asymmetric singularity subtraction.  Convolution time: ', convolutionTime)
#                     return
                elif GPUpresent==True:
                    if treecode==False:
                        startTime = time.time()
                        if symmetricIteration==False:
    
                            temp=np.transpose( np.array([X,Y,Z,f,W]) )
                            gpuHelmholtzConvolutionSubractSingularity[blocksPerGrid, threadsPerBlock](temp,temp,phiNew,k) 
                            convolutionTime = time.time()-startTime
                            print('Using asymmetric singularity subtraction.  Convolution time: ', convolutionTime)
                        elif symmetricIteration==True:
                            gpuHelmholtzConvolutionSubractSingularitySymmetric[blocksPerGrid, threadsPerBlock](targets,sources,sqrtV,phiNew,k) 
                            phiNew *= -1
                            convolutionTime = time.time()-startTime
                            print('Using symmetric singularity subtraction.  Convolution time: ', convolutionTime)
                        convTime=time.time()-startTime
                        print('Convolution time: ', convTime)
                        Times['timePerConvolution'] = convTime
                        
                    elif treecode==True:
                        
    
                        potentialType=3
                        kappa = k
                        startTime = time.time()
                        numDevices=4
                        numThreads=4
                        phiNew = treecodeWrappers.callTreedriver(nPoints, nPoints, 
                                                                       np.copy(X), np.copy(Y), np.copy(Z), np.copy(f), 
                                                                       np.copy(X), np.copy(Y), np.copy(Z), np.copy(f), np.copy(W),
                                                                       potentialType, kappa, treecodeOrder, theta, maxParNode, batchSize, numDevices, numThreads)
                    
    
                        convTime=time.time()-startTime
                        print('Convolution time: ', convTime)
                        Times['timePerConvolution'] = convTime
                        phiNew /= (4*np.pi)
                    
                    else: 
                        print('treecode true or false?')
                        return
            else:
                print('Using singularity skipping because energy too close to 0')
                gpuHelmholtzConvolution[blocksPerGrid, threadsPerBlock](np.array([X,Y,Z,f,W]),np.array([X,Y,Z,f,W]),phiNew,k)
        else:
            print('Invalid option for singularitySubtraction, should be 0 or 1.')
            return
        

        
        
        # update the energy first
        
        psiNewNorm = np.sqrt( np.sum( phiNew*phiNew*W))
        Energies['orbitalEnergies'][m] = psiNewNorm
        print('Norm of psiNew = ', psiNewNorm)
        orbitals[:,m] = np.copy(phiNew)


        

        n,M = np.shape(orbitals) 
        orthWavefunction = mgs(orbitals,W,m, n, M)
        
        orbitals[:,m] = np.copy(orthWavefunction)
        
 

        if greenIterationsCount==1:
            eigenvalueHistory = np.array(Energies['orbitalEnergies'][m])
        else:
            eigenvalueHistory = gi_args['eigenvalueHistory']
            eigenvalueHistory = np.append(eigenvalueHistory, Energies['orbitalEnergies'][m])
        print('eigenvalueHistory: \n',eigenvalueHistory)
        
        
        print('Orbital energy after Harrison update: ', Energies['orbitalEnergies'][m])
             
    

        psiOut = np.append(np.copy(orbitals[:,m]), np.copy(Energies['orbitalEnergies'][m]))
        residualVector = psiOut - psiIn
        
        
        
        loc = np.argmax(np.abs(residualVector[:-1]))
    
        newEigenvalue = Energies['orbitalEnergies'][m]
        
        
            
    
        
        if symmetricIteration==False:
            normDiff = np.sqrt( np.sum( (orbitals[:,m]-oldOrbitals[:,m])**2*W ) )
        elif symmetricIteration==True:
            normDiff = np.sqrt( np.sum( (orbitals[:,m]*sqrtV-oldOrbitals[:,m]*sqrtV)**2*W ) )
        eigenvalueDiff = abs(newEigenvalue - oldEigenvalue)    
        
    
        residuals[m] = normDiff
        orbitalResidual = np.copy(normDiff)
        
        
    
#         print('Orbital %i error and eigenvalue residual:   %1.3e and %1.3e' %(m,Energies['orbitalEnergies'][m]-referenceEigenvalues[m]-Energies['gaugeShift'], eigenvalueDiff))
        print('Orbital %i eigenvalue residual:  %1.3e' %(m,eigenvalueDiff))
        print('Orbital %i wavefunction residual: %1.3e' %(m, orbitalResidual))
        print()
        print()
    
    
    
        header = ['targetOrbital', 'Iteration', 'orbitalResiduals', 'energyEigenvalues', 'eigenvalueResidual']
    
        myData = [m, greenIterationsCount, residuals,
                  Energies['orbitalEnergies'], eigenvalueDiff]
    
        if not os.path.isfile(greenIterationOutFile):
            myFile = open(greenIterationOutFile, 'a')
            with myFile:
                writer = csv.writer(myFile)
                writer.writerow(header) 
            
        
        myFile = open(greenIterationOutFile, 'a')
        with myFile:
            writer = csv.writer(myFile)
            writer.writerow(myData)
        
          
        greenIterationsCount += 1
#         greenIterationsCount += 1   
        
        
        # Pack up gi_args (only things that get modified by the call to this function)
#         gi_args_out = np.copy(gi_args)
#         gi_args_out['orbitals'] = orbitals
#         gi_args_out['oldOrbitals'] = oldOrbitals
#         gi_args_out['Energies'] = Energies
#         gi_args_out['Times'] = Times
#         gi_args_out['greenIterationsCount'] = greenIterationsCount
#         gi_args_out['residuals'] = residuals
#         gi_args_out['eigenvalueDiff']=eigenvalueDiff
        
        
        gi_args['orbitals'] = orbitals
        gi_args['oldOrbitals'] = oldOrbitals
        gi_args['Energies'] = Energies
        gi_args['Times'] = Times
        gi_args['greenIterationsCount'] = greenIterationsCount
        gi_args['residuals'] = residuals
        gi_args['eigenvalueDiff']=eigenvalueDiff
        gi_args['eigenvalueHistory']=eigenvalueHistory
        
        
        
        return residualVector
    
    return eigenvalueOne_FixedPoint, gi_args 