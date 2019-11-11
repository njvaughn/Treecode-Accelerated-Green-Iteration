import inspect
import numpy as np
import time
import csv
import resource
import GPUtil
import os

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

from mpiUtilities import global_dot, rprint


 


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
    import treecodeWrappers_distributed as treecodeWrappers
except ImportError:
    print('Unable to import treecodeWrapper due to ImportError')
except OSError:
    print('Unable to import treecodeWrapper due to OSError')
    

# try:
#     from orthogonalizationRoutines import *
# except ImportError:
from orthogonalizationRoutines import modifiedGramSchmidt_singleOrbital as mgs


def greensIteration_FixedPoint_Closure(gi_args):
#     gi_args_out = {}
    def greensIteration_FixedPoint(psiIn, gi_args):
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
        singularityHandling = gi_args['singularityHandling']
        approximationName = gi_args['approximationName']
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
#         print("X, Y, Z, W: ", X[:3], Y[:3], Z[:3], W[:3])
        gradientFree = gi_args['gradientFree']
        SCFcount = gi_args['SCFcount']
        greenIterationsCount = gi_args['greenIterationsCount']
        residuals = gi_args['residuals']
        greenIterationOutFile = gi_args['greenIterationOutFile']
        threadsPerBlock=gi_args['threadsPerBlock']
        blocksPerGrid=gi_args['blocksPerGrid']
        referenceEigenvalues = gi_args['referenceEigenvalues']
        updateEigenvalue = gi_args['updateEigenvalue']

        
#         print('Who called F(x)? ', inspect.stack()[2][3])
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
        
    
     
        if symmetricIteration==False:
            f = -2*orbitals[:,m]*Veff
        else: 
            print("symmetricIteration variable not True or False.  What should it be?")
            return
        
        
        oldEigenvalue =  Energies['orbitalEnergies'][m] 
        k = np.sqrt(-2*Energies['orbitalEnergies'][m])
#         print('k = ', k)
    
        phiNew = np.zeros(nPoints)
        
        if Energies['orbitalEnergies'][m] < 10.25**100: 
            
            
            if GPUpresent==False:
                startTime=time.time()
#                 potentialType=3
                kernelName = "yukawa"
#                 potentialType=1
#                 print('potentialType=1')
                kappa = k
                startTime = time.time()
                numDevices=gi_args['numDevices']
                numThreads=gi_args['numThreads']
                comm.barrier()
                phiNew = treecodeWrappers.callTreedriver(nPoints, nPoints, 
                                                               np.copy(X), np.copy(Y), np.copy(Z), np.copy(f), 
                                                               np.copy(X), np.copy(Y), np.copy(Z), np.copy(f), np.copy(W),
                                                               kernelName, kappa, singularityHandling, approximationName, treecodeOrder, theta, maxParNode, batchSize, GPUpresent)
#                 print("Length of phiNew: ", len(phiNew))
#                 print("Max of phiNew: ", np.max(np.abs(phiNew)))
                phiNew += 4*np.pi*f/k**2
                if singularityHandling=="skipping": phiNew /= (4*np.pi)
                if singularityHandling=="subtraction": phiNew /= (4*np.pi)
#                 print("Max of phiNew: ", np.max(np.abs(phiNew)))
#                 print("Avg of phiNew: ", np.mean(phiNew))
                convolutionTime = time.time()-startTime
                rprint('Using asymmetric singularity subtraction.  Convolution time: ', convolutionTime)
                comm.barrier()
#                     return
            elif GPUpresent==True:
                if treecode==False:
                    startTime = time.time()
                    if symmetricIteration==False:

                        temp=np.transpose( np.array([X,Y,Z,f,W]) )
                        if subtractSingularity==0:
                            print("Using singularity skipping in Greens iteration direct sum.")
                            gpuHelmholtzConvolution[blocksPerGrid, threadsPerBlock](temp,temp,phiNew,k) 
                        else:
                            gpuHelmholtzConvolutionSubractSingularity[blocksPerGrid, threadsPerBlock](temp,temp,phiNew,k) 
                        convolutionTime = time.time()-startTime
#                         print('Using asymmetric singularity subtraction.  Convolution time: ', convolutionTime)
                    elif symmetricIteration==True:
                        gpuHelmholtzConvolutionSubractSingularitySymmetric[blocksPerGrid, threadsPerBlock](targets,sources,sqrtV,phiNew,k) 
                        phiNew *= -1
                        convolutionTime = time.time()-startTime
                        print('Using symmetric singularity subtraction.  Convolution time: ', convolutionTime)
                    convTime=time.time()-startTime
                    print('Convolution time: ', convTime)
                    Times['timePerConvolution'] = convTime  
                    
                    
                    
                elif treecode==True:
                    
                    if subtractSingularity==0:
                        print("Using singularity skipping in Green's iteration.")
                        potentialType=3
                        kernelName="yukawa_SS"
                    else: 
                        potentialType=3
                        kernelName="yukawa_SS"
                    kappa = k
                    startTime = time.time()
                    numDevices=gi_args['numDevices']
                    numThreads=gi_args['numThreads']
                    
#                     print('numDevices and numThreads as read in from gi_args are: ', numDevices, numThreads)
                    phiNew = treecodeWrappers.callTreedriver(nPoints, nPoints, 
                                                                   np.copy(X), np.copy(Y), np.copy(Z), np.copy(f), 
                                                                   np.copy(X), np.copy(Y), np.copy(Z), np.copy(f), np.copy(W),
                                                                   kernelName, kappa, treecodeOrder, theta, maxParNode, batchSize, GPUpresent)
                

                    if subtractSingularity==1: phiNew /= (4*np.pi)
                    convTime=time.time()-startTime
                    print('Convolution time: ', convTime)
                    Times['timePerConvolution'] = convTime
                    
                
                else: 
                    print('treecode true or false?')
                    return
        else:
            print('Using singularity skipping because energy too close to 0')
            gpuHelmholtzConvolution[blocksPerGrid, threadsPerBlock](np.array([X,Y,Z,f,W]),np.array([X,Y,Z,f,W]),phiNew,k)
        
        
#         print('Max phiNew: ', np.max(phiNew))
#         print('Min phiNew: ', np.min(phiNew))
        
        """ Method where you dont compute kinetics, from Harrison """
        if updateEigenvalue==True:
            # update the energy first
            
            orthWavefunction2 = np.zeros(nPoints)
            if ( (gradientFree==True) and (SCFcount>-1)):                 
                
#                 psiNewNorm = np.sqrt( np.sum( phiNew*phiNew*W))
                psiNewNorm = np.sqrt( global_dot( phiNew, phiNew*W, comm))
                rprint("psiNewNorm = %f" %psiNewNorm)
                
                if symmetricIteration==False:
        
#                     deltaE = -np.sum( orbitals[:,m]*Veff*(orbitals[:,m]-phiNew)*W ) 
                    deltaE = -global_dot( orbitals[:,m]*Veff*(orbitals[:,m]-phiNew), W, comm ) 
#                     normSqOfPsiNew = np.sum( phiNew**2 * W)
                    normSqOfPsiNew = global_dot( phiNew**2, W, comm)
                    deltaE /= (normSqOfPsiNew)  # divide by norm squared, according to Harrison-Fann- et al
    #                 deltaE /= (psiNewNorm)
#                     print('NormSq of psiNew = ', normSqOfPsiNew )
#                     print('Norm of psiNew = ', psiNewNorm )
#                     print('Delta E = ', deltaE)
                    rprint("deltaE = %f" %deltaE)
                    Energies['orbitalEnergies'][m] += deltaE
                    rprint("Energies['orbitalEnergies'][m] = %f" %Energies['orbitalEnergies'][m])
                    orbitals[:,m] = np.copy(phiNew)
                elif symmetricIteration==True:
                    print('Symmetric not set up for tree-free')
                    return
        
                n,M = np.shape(orbitals) 
    #             Wcopy = np.copy(W)
    #             mcopy = np.copy(m)
    #             nPointsCopy = np.copy(nPoints)
                orthWavefunction = mgs(orbitals,W,m, n, M, comm)
    #             modifiedGramSchmidt_singleOrbital_GPU[blocksPerGrid, threadsPerBlock](np.copy(orbitals),Wcopy,mcopy,nPointsCopy, orthWavefunction2)
                
                orbitals[:,m] = np.copy(orthWavefunction)
    #             orbitals[:,m] = np.copy(orthWavefunction2)
        #         tree.importPhiOnLeaves(orbitals[:,m], m)
                
         
        
                if greenIterationsCount==1:
                    eigenvalueHistory = np.array(Energies['orbitalEnergies'][m])
                else:
                    eigenvalueHistory = gi_args['eigenvalueHistory']
                    eigenvalueHistory = np.append(eigenvalueHistory, Energies['orbitalEnergies'][m])
                rprint('eigenvalueHistory: \n',eigenvalueHistory)
                
                
#                 print('Orbital energy after Harrison update: ', Energies['orbitalEnergies'][m])
                 
        
        #     elif ( (gradientFree==False) or (SCFcount==-1) and False ):
            elif ( (gradientFree==False) or (gradientFree=='Laplacian') ):
                print('gradient and laplacian methods not set up for tree-free')
                return
                
            else:
                print('Not updating eigenvalue.  Is that intended?')
                return
            
        else:  # Explicitly choosing to not update Eigenvalue.  Still orthogonalize
            print("Not updating eigenvalue because updateEigenvalue!=True")
            orbitals[:,m] = np.copy(phiNew)
            n,M = np.shape(orbitals) 
            orthWavefunction = mgs(orbitals,W,m, n, M, comm)
            orbitals[:,m] = np.copy(orthWavefunction) 
            if greenIterationsCount==1:
                eigenvalueHistory = np.array(Energies['orbitalEnergies'][m])
            else:
                eigenvalueHistory = gi_args['eigenvalueHistory']
                eigenvalueHistory = np.append(eigenvalueHistory, Energies['orbitalEnergies'][m])
            if rank==0: print('eigenvalueHistory (should be constant): \n',eigenvalueHistory)
            gi_args['eigenvalueDiff']=0
            deltaE=0
            gi_args['eigenvalueHistory']=eigenvalueHistory
            
    
        
        if Energies['orbitalEnergies'][m]>0.0:
#         if Energies['orbitalEnergies'][m]>Energies['gaugeShift']:
#             Energies['orbitalEnergies'][m] = Energies['gaugeShift'] - np.random.randint(10)
            rand = np.random.rand(1)
            Energies['orbitalEnergies'][m] = Energies['gaugeShift'] - 3*rand
            print('Energy eigenvalue was positive, setting to gauge shift - ', 3*rand)
            
            if greenIterationsCount%10==0:
                # Positive energy after 10 iterations..., scramble wavefunction and restart.
                orbitals[:,m] = np.ones(len(np.copy(orbitals[:,m])))
            
        
    #     tree.printWavefunctionNearEachAtom(m)
            
    #     residualVector = orbitals[:,m] - oldOrbitals[:,m]
        psiOut = np.append(np.copy(orbitals[:,m]), np.copy(Energies['orbitalEnergies'][m]))
        residualVector = psiOut - psiIn
        
        
        
#         print('Max value of wavefunction: ', np.max(np.abs(orbitals[:,m])))
        loc = np.argmax(np.abs(residualVector[:-1]))
#         print('Largest residual: ', residualVector[loc])
#         print('Value at that point: ', psiOut[loc])
#         print('Location of max residual: ', X[loc], Y[loc], Z[loc])
    #     residualVector = -(psiIn - orbitals[:,m])
    
        newEigenvalue = Energies['orbitalEnergies'][m]
        
        
            
    
        
        if symmetricIteration==False:
#             normDiff = np.sqrt( np.sum( (orbitals[:,m]-oldOrbitals[:,m])**2*W ) )
            normDiff = np.sqrt( global_dot( (orbitals[:,m]-oldOrbitals[:,m])**2,W, comm ) )
        elif symmetricIteration==True:
#             normDiff = np.sqrt( np.sum( (orbitals[:,m]*sqrtV-oldOrbitals[:,m]*sqrtV)**2*W ) )
            normDiff = np.sqrt( global_dot( (orbitals[:,m]*sqrtV-oldOrbitals[:,m]*sqrtV)**2,W,comm ) )
        eigenvalueDiff = abs(newEigenvalue - oldEigenvalue)    
        
        if rank==0:
            residuals[m] = normDiff
        orbitalResidual = np.copy(normDiff)
        
        
    
        rprint('Orbital %i error and eigenvalue residual:   %1.3e and %1.3e' %(m,Energies['orbitalEnergies'][m]-referenceEigenvalues[m]-Energies['gaugeShift'], eigenvalueDiff))
        rprint('Orbital %i wavefunction residual: %1.3e\n\n' %(m, orbitalResidual))
    
    
    
        header = ['targetOrbital', 'Iteration', 'orbitalResiduals', 'energyEigenvalues', 'eigenvalueResidual']
    
        myData = [m, greenIterationsCount, residuals,
                  Energies['orbitalEnergies']-Energies['gaugeShift'], eigenvalueDiff]
    
        if rank==0:
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
        gi_args['eigenvalueDiff'] = np.abs(deltaE)
        
        
        
        return residualVector
    
    return greensIteration_FixedPoint, gi_args 