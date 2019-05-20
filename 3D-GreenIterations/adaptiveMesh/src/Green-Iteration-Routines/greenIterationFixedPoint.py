import inspect
import numpy as np
import time
import csv

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
    

from orthogonalizationRoutines import *



def greensIteration_FixedPoint_Closure(gi_args):
#     gi_args_out = {}
    def greensIteration_FixedPoint(psiIn, gi_args):
        # what other things do we need?  Energies, Times, orbitals, Veff, runtime constants (symmetricIteration, GPUpresent, subtractSingularity, treecode, outputfiles, ...)  
        
        
        ## UNPACK GIARGS
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
        n,M = np.shape(orbitals)
        orthWavefunction = modifiedGramSchmidt_singleOrbital(orbitals,W,m, n, M)
        orbitals[:,m] = np.copy(orthWavefunction)
        Energies['orbitalEnergies'][m] = np.copy(psiIn[-1])
        
    
     
        if symmetricIteration==False:
            f = -2*orbitals[:,m]*Veff
        else: 
            print("symmetricIteration variable not True or False.  What should it be?")
            return
        
        
        oldEigenvalue =  Energies['orbitalEnergies'][m] 
        k = np.sqrt(-2*Energies['orbitalEnergies'][m])
        print('k = ', k)
    
        phiNew = np.zeros(nPoints)
        if subtractSingularity==0: 
            print('Using singularity skipping')
            gpuHelmholtzConvolution[blocksPerGrid, threadsPerBlock](targets,sources,phiNew,k) 
        elif subtractSingularity==1:
            if Energies['orbitalEnergies'][m] < 10.25**100: 
                
                
                if GPUpresent==False:
                    print('No GPU?')
                    return
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
                        phiNew = treecodeWrappers.callTreedriver(nPoints, nPoints, 
                                                                       np.copy(X), np.copy(Y), np.copy(Z), np.copy(f), 
                                                                       np.copy(X), np.copy(Y), np.copy(Z), np.copy(f), np.copy(W),
                                                                       potentialType, kappa, treecodeOrder, theta, maxParNode, batchSize)
                    
    
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
        
        print('Max phiNew: ', np.max(phiNew))
        print('Min phiNew: ', np.min(phiNew))
        
        """ Method where you dont compute kinetics, from Harrison """
        
        # update the energy first
        
    
        if ( (gradientFree==True) and (SCFcount>-1)):                 
            
            psiNewNorm = np.sqrt( np.sum( phiNew*phiNew*W))
            
            if symmetricIteration==False:
    
                deltaE = -np.sum( orbitals[:,m]*Veff*(orbitals[:,m]-phiNew)*W ) 
                normSqOfPsiNew = np.sum( phiNew**2 * W)
                deltaE /= (normSqOfPsiNew)
                print('Norm of psiNew = ', np.sqrt(normSqOfPsiNew))
                print('Delta E = ', deltaE)
                Energies['orbitalEnergies'][m] += deltaE
                orbitals[:,m] = np.copy(phiNew)
            elif symmetricIteration==True:
                print('Symmetric not set up for tree-free')
                return
    
            n,M = np.shape(orbitals)
            orthWavefunction = modifiedGramSchmidt_singleOrbital(orbitals,W,m, n, M)
            orbitals[:,m] = np.copy(orthWavefunction)
    #         tree.importPhiOnLeaves(orbitals[:,m], m)
            
    
    
            if greenIterationsCount==1:
                eigenvalueHistory = np.array(Energies['orbitalEnergies'][m])
            else:
                eigenvalueHistory = gi_args['eigenvalueHistory']
                eigenvalueHistory = np.append(eigenvalueHistory, Energies['orbitalEnergies'][m])
            print('eigenvalueHistory: \n',eigenvalueHistory)
            
            
            print('Orbital energy after Harrison update: ', Energies['orbitalEnergies'][m])
             
    
    #     elif ( (gradientFree==False) or (SCFcount==-1) and False ):
        elif ( (gradientFree==False) or (gradientFree=='Laplacian') ):
            print('gradient and laplacian methods not set up for tree-free')
            return
            
        else:
            print('Not updating eigenvalue.  Is that intended?')
            return
    
        
        if Energies['orbitalEnergies'][m]>0.0:
            Energies['orbitalEnergies'][m] = Energies['gaugeShift'] - 0.5
            print('Energy eigenvalue was positive, setting to gauge shift - 0.5')
            
        
    #     tree.printWavefunctionNearEachAtom(m)
            
    #     residualVector = orbitals[:,m] - oldOrbitals[:,m]
        psiOut = np.append(np.copy(orbitals[:,m]), np.copy(Energies['orbitalEnergies'][m]))
        residualVector = psiOut - psiIn
        
        
        
        loc = np.argmax(np.abs(residualVector[:-1]))
        print('Largest residual: ', residualVector[loc])
        print('Value at that point: ', psiOut[loc])
        print('Location of max residual: ', X[loc], Y[loc], Z[loc])
    #     residualVector = -(psiIn - orbitals[:,m])
    
        newEigenvalue = Energies['orbitalEnergies'][m]
        
        
            
    
        
        if symmetricIteration==False:
            normDiff = np.sqrt( np.sum( (orbitals[:,m]-oldOrbitals[:,m])**2*W ) )
        elif symmetricIteration==True:
            normDiff = np.sqrt( np.sum( (orbitals[:,m]*sqrtV-oldOrbitals[:,m]*sqrtV)**2*W ) )
        eigenvalueDiff = abs(newEigenvalue - oldEigenvalue)    
        
    
        residuals[m] = normDiff
        orbitalResidual = np.copy(normDiff)
        
        
    
        print('Orbital %i error and eigenvalue residual:   %1.3e and %1.3e' %(m,Energies['orbitalEnergies'][m]-referenceEigenvalues[m]-Energies['gaugeShift'], eigenvalueDiff))
        print('Orbital %i wavefunction residual: %1.3e' %(m, orbitalResidual))
        print()
        print()
    
    
    
        header = ['targetOrbital', 'Iteration', 'orbitalResiduals', 'energyEigenvalues', 'eigenvalueResidual']
    
        myData = [m, greenIterationsCount, residuals,
                  Energies['orbitalEnergies']-Energies['gaugeShift'], eigenvalueDiff]
    
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
    
    return greensIteration_FixedPoint, gi_args