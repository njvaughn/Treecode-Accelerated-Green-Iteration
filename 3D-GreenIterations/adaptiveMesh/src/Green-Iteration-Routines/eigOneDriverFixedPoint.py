import numpy as np
import os
import csv
from numba import cuda, jit, njit
import time
# from scipy.optimize import anderson as scipyAnderson
from scipy.optimize import root as scipyRoot
from scipy.optimize.nonlin import BroydenFirst, KrylovJacobian
from scipy.optimize.nonlin import InverseJacobian
from scipy.optimize import broyden1, anderson, brentq
# from scipy.optimize import newton_krylov as scipyNewtonKrylov
import GPUtil

from fermiDiracDistribution import computeOccupations
import densityMixingSchemes as densityMixing
import sys
import resource
# sys.path.append(srcdir+'../ctypesTests/src')
# 
# sys.path.append(srcdir+'../ctypesTests')
# sys.path.append(srcdir+'../ctypesTests/lib') 

import treecodeWrappers
# from greenIterationFixedPoint import greensIteration_FixedPoint_Closure
from eigenvalueOneFixedPoint import eigenvalueOne_FixedPoint_Closure
from orthogonalizationRoutines import *
try:
    from convolution import *
except ImportError:
    print('Unable to import JIT GPU Convolutions')
# try:
#     import directSumWrappers
# except ImportError:
#     print('Unable to import directSumWrappers due to ImportError')
# except OSError:
#     print('Unable to import directSumWrappers due to OSError')
#     
# try:
#     import treecodeWrappers
# except ImportError:
#     print('Unable to import treecodeWrapper due to ImportError')
# except OSError:
#     print('Unable to import treecodeWrapper due to OSError')


Temperature = 200
KB = 1/315774.6
Sigma = Temperature*KB
def fermiObjectiveFunctionClosure(Energies,nElectrons):
    def fermiObjectiveFunction(fermiEnergy):
                exponentialArg = (Energies['orbitalEnergies']-fermiEnergy)/Sigma
                temp = 1/(1+np.exp( exponentialArg ) )
                return nElectrons - 2 * np.sum(temp)
    return fermiObjectiveFunction


def clenshawCurtisNormClosure(W):
    def clenshawCurtisNorm(psi):
        appendedWeights = np.append(W, 1.0)
        norm = np.sqrt( np.sum( psi*psi*appendedWeights ) )
#         norm = np.sqrt( np.sum( psi[-1]*psi[-1]*appendedWeights[-1] ) )
        return norm
    return clenshawCurtisNorm

def printResidual(x,f):
    r = clenshawCurtisNorm(f)
    print('L2 Norm of Residual: ', r)
    
def sortByEigenvalue(orbitals,orbitalEnergies):
    newOrder = np.argsort(orbitalEnergies)
    oldEnergies = np.copy(orbitalEnergies)
    for m in range(len(orbitalEnergies)):
        orbitalEnergies[m] = oldEnergies[newOrder[m]]
    print('Sorted eigenvalues: ', orbitalEnergies)
    print('New order: ', newOrder)
    
    newOrbitals = np.zeros_like(orbitals)
    for m in range(len(orbitalEnergies)):
        newOrbitals[:,m] = orbitals[:,newOrder[m]]
#         if newOrder[m]!=m:
            
   
    return newOrbitals, orbitalEnergies

def sortLargestFirst(orbitals,orbitalEnergies):
    newOrder = np.argsort(orbitalEnergies)[::-1] ## [::-1] to get largest first....
    oldEnergies = np.copy(orbitalEnergies)
    for m in range(len(orbitalEnergies)):
        orbitalEnergies[m] = oldEnergies[newOrder[m]]
    print('Sorted eigenvalues: ', orbitalEnergies)
    print('New order: ', newOrder)
    
    newOrbitals = np.zeros_like(orbitals)
    for m in range(len(orbitalEnergies)):
        newOrbitals[:,m] = orbitals[:,newOrder[m]]
#         if newOrder[m]!=m:
            
   
    return newOrbitals, orbitalEnergies  
    
def eigOneDriverFixedPointClosure(scf_args): 
    
    def eigOneDriverFixedPoint(RHO,targetEpsilon,scf_args):
        
        ## Unpack scf_args
        inputDensities = scf_args['inputDensities']
        outputDensities=scf_args['outputDensities']
        SCFcount = scf_args['SCFcount']
        nPoints = scf_args['nPoints']
        nMu=scf_args['nMu']
        nElectrons=scf_args['nElectrons']
        mixingHistoryCutoff = scf_args['mixingHistoryCutoff']
        GPUpresent = scf_args['GPUpresent']
        treecode = scf_args['treecode']
        treecodeOrder=scf_args['treecodeOrder']
        theta=scf_args['theta']
        maxParNode=scf_args['maxParNode']
        batchSize=scf_args['batchSize']
        gaussianAlpha=scf_args['gaussianAlpha']
        Energies=scf_args['Energies']
        exchangeFunctional=scf_args['exchangeFunctional']
        correlationFunctional=scf_args['correlationFunctional']
        Vext=scf_args['Vext']
        gaugeShift=scf_args['gaugeShift']
        orbitals=scf_args['orbitals']
        oldOrbitals=scf_args['oldOrbitals']
        Times=scf_args['Times']
        subtractSingularity=scf_args['subtractSingularity']
        X = scf_args['X']
        Y = scf_args['Y']
        Z = scf_args['Z']
        W = scf_args['W']
        gradientFree = scf_args['gradientFree']
        residuals = scf_args['residuals']
        greenIterationOutFile = scf_args['greenIterationOutFile']
        threadsPerBlock=scf_args['threadsPerBlock']
        blocksPerGrid=scf_args['blocksPerGrid']
        referenceEigenvalues = scf_args['referenceEigenvalues']
        symmetricIteration=scf_args['symmetricIteration']
        intraScfTolerance=scf_args['intraScfTolerance']
        referenceEnergies=scf_args['referenceEnergies']
        SCFiterationOutFile=scf_args['SCFiterationOutFile']
        wavefunctionFile=scf_args['wavefunctionFile']
        densityFile=scf_args['densityFile']
        outputDensityFile=scf_args['outputDensityFile']
        inputDensityFile=scf_args['inputDensityFile']
        vHartreeFile=scf_args['vHartreeFile']
        auxiliaryFile=scf_args['auxiliaryFile']
        
        
        
        
        
        SCFcount += 1
        print()
        print()
        print('\nSCF Count ', SCFcount)
        print('Orbital Energies: ', Energies['orbitalEnergies'])
        
        
#         if SCFcount>1:
#             
#     
#             if (SCFcount-1)<mixingHistoryCutoff:
#                 inputDensities = np.concatenate( (inputDensities, np.reshape(RHO, (nPoints,1))), axis=1)
#                 print('Concatenated inputDensity.  Now has shape: ', np.shape(inputDensities))
#             else:
#                 print('Beyond mixingHistoryCutoff.  Replacing column ', (SCFcount-1)%mixingHistoryCutoff)
#     #                                 print('Shape of oldOrbitals[:,m]: ', np.shape(oldOrbitals[:,m]))
#                 inputDensities[:,(SCFcount-1)%mixingHistoryCutoff] = np.copy(RHO)
        
     
        
    
        ### Compute Veff
        """ 
        Compute new electron-electron potential and update pointwise potential values 
        """
    #         starthartreeConvolutionTime = timer()
        
        if SCFcount>-1: # need to compute Hartree potential at start
            if GPUpresent==True:
                if treecode==False:
                    V_hartreeNew = np.zeros(nPoints)
                    densityInput = np.transpose( np.array([X,Y,Z,RHO,W]) )
                    gpuHartreeGaussianSingularitySubract[blocksPerGrid, threadsPerBlock](densityInput,densityInput,V_hartreeNew,gaussianAlpha*gaussianAlpha)
                elif treecode==True:
                    start = time.time()
                    potentialType=2 
                    numThreads=4
                    numDevices=4
                    V_hartreeNew = treecodeWrappers.callTreedriver(nPoints, nPoints, 
                                                                   np.copy(X), np.copy(Y), np.copy(Z), np.copy(RHO), 
                                                                   np.copy(X), np.copy(Y), np.copy(Z), np.copy(RHO), np.copy(W),
                                                                   potentialType, gaussianAlpha, treecodeOrder, theta, maxParNode, batchSize, numDevices, numThreads)
                    print('Convolution time: ', time.time()-start)
                    
            elif GPUpresent==False:
                if treecode==False:
                    V_hartreeNew = np.zeros(nPoints)
                    densityInput = np.transpose( np.array([X,Y,Z,RHO,W]) )
                    gpuHartreeGaussianSingularitySubract(densityInput,densityInput,V_hartreeNew,gaussianAlpha*gaussianAlpha)
                else:    
                    potentialType=2 
                    numThreads=4
                    numDevices=0
                    V_hartreeNew = treecodeWrappers.callTreedriver(nPoints, nPoints, 
                                                                   np.copy(X), np.copy(Y), np.copy(Z), np.copy(RHO), 
                                                                   np.copy(X), np.copy(Y), np.copy(Z), np.copy(RHO), np.copy(W),
                                                                   potentialType, gaussianAlpha, treecodeOrder, theta, maxParNode, batchSize, numDevices, numThreads)
            else:
                print('Is GPUpresent supposed to be true or false?')
                return
      
        
        
        """ 
        Compute the new orbital and total energies 
        """
        
        ## Energy update after computing Vhartree
        
            
        Energies['Ehartree'] = 1/2*np.sum(W * RHO * V_hartreeNew)
    
    
        exchangeOutput = exchangeFunctional.compute(RHO)
        correlationOutput = correlationFunctional.compute(RHO)
        Energies['Ex'] = np.sum( W * RHO * np.reshape(exchangeOutput['zk'],np.shape(RHO)) )
        Energies['Ec'] = np.sum( W * RHO * np.reshape(correlationOutput['zk'],np.shape(RHO)) )
        
        Vx = np.reshape(exchangeOutput['vrho'],np.shape(RHO))
        Vc = np.reshape(correlationOutput['vrho'],np.shape(RHO))
        
        Energies['Vx'] = np.sum(W * RHO * Vx)
        Energies['Vc'] = np.sum(W * RHO * Vc)
        
        Veff = V_hartreeNew + Vx + Vc + Vext + gaugeShift
        
        if SCFcount==1: # generate initial guesses for eigenvalues
            Energies['Eold']=-10
            for m in range(nMu):
#                 Energies['orbitalEnergies'][m] = -10
                Energies['orbitalEnergies'][m] = np.sum( W* orbitals[:,m]**2 * Veff) * (2/3) # Attempt to guess initial orbital energy without computing kinetic
            orbitals, Energies['orbitalEnergies'] = sortByEigenvalue(orbitals, Energies['orbitalEnergies'])
            for m in range(nMu):
                if Energies['orbitalEnergies'][m] > 0:
                    Energies['orbitalEnergies'][m] = -0.5
        
        
           
        
        ## Solve the eigenvalue problem
        
        for m in range(nMu): 
            print('Working on orbital %i' %m)
            print('MEMORY USAGE: ', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss )
            
    #                         
                        
            greenIterationsCount=1
            gi_args = {'orbitals':orbitals,'oldOrbitals':oldOrbitals, 'Energies':Energies, 'Times':Times, 'Veff':Veff, 
                           'symmetricIteration':symmetricIteration,'GPUpresent':GPUpresent,'subtractSingularity':subtractSingularity,
                           'treecode':treecode,'treecodeOrder':treecodeOrder,'theta':theta, 'maxParNode':maxParNode,'batchSize':batchSize,
                           'nPoints':nPoints, 'm':m, 'X':X,'Y':Y,'Z':Z,'W':W,'gradientFree':gradientFree,
                           'SCFcount':SCFcount,'greenIterationsCount':greenIterationsCount,'residuals':residuals,
                           'greenIterationOutFile':greenIterationOutFile, 'blocksPerGrid':blocksPerGrid,'threadsPerBlock':threadsPerBlock,
                           'referenceEigenvalues':referenceEigenvalues, 'targetEpsilon':targetEpsilon   } 
            
            n,M = np.shape(orbitals)
            resNorm=1 
            
            orthWavefunction = modifiedGramSchmidt_singleOrbital(orbitals,W,m, n, M)
            orbitals[:,m] = np.copy(orthWavefunction)
            
            while ( (resNorm>intraScfTolerance) or (gi_args["greenIterationsCount"]<5) ):
#             while resNorm>intraScfTolerance:
#                 print('MEMORY USAGE: ', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss )
#                 GPUtil.showUtilization()
#                 print('MEMORY USAGE: ', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss )
            
                # Orthonormalize orbital m before beginning Green's iteration
#                 n,M = np.shape(orbitals)
#                 orthWavefunction = modifiedGramSchmidt_singleOrbital(orbitals,W,m, n, M)
#                 orbitals[:,m] = np.copy(orthWavefunction)
#                 print('orbitals before: ',orbitals[1:5,m])
#                 print('greenIterationsCount before: ', greenIterationsCount)
                psiIn = np.append( np.copy(orbitals[:,m]), Energies['orbitalEnergies'][m] )
#                 print('psiIn = ', psiIn[:5])
        
        
                eigOne_FixedPoint, gi_args = eigenvalueOne_FixedPoint_Closure(gi_args)
                r = eigOne_FixedPoint(psiIn, gi_args)
#                 print('r = ', r[:5])
                psiOut = np.append(orbitals[:,m],Energies['orbitalEnergies'][m])
#                 print('greenIterationsCount after: ', greenIterationsCount)
#                 print('gi_args greenIterationsCount after: ', gi_args["greenIterationsCount"])

#                 print('psiOut: ',psiOut[:5])
#                 print('gi_args orbitals after: ',gi_args["orbitals"][1:5,m])
                clenshawCurtisNorm = clenshawCurtisNormClosure(W)
                resNorm = clenshawCurtisNorm(psiOut-psiIn)
                print('CC norm of residual vector: ', resNorm)
                eigenvalueDiff = np.abs(psiOut[-1] - psiIn[-1])
                
                if eigenvalueDiff < intraScfTolerance/10:
                    resNorm=0.0  

            
            psiOut = np.append(orbitals[:,m],Energies['orbitalEnergies'][m])
            print('Power iteration tolerance met.  Beginning rootfinding now...') 
#             psiIn = np.copy(psiOut)
            tol=intraScfTolerance
            
            
            ## Begin Anderson Mixing on Wavefunction
#             Done = False
# #             Done = True
#             firstInputWavefunction=True
#             firstOutputWavefunction=True
#             inputWavefunctions = np.zeros((psiOut.size,1))
#             outputWavefunctions =  np.zeros((psiOut.size,1))
#             mixingStart = np.copy( gi_args['greenIterationsCount'] )
#             while Done==False:
#                 greenIterationsCount=gi_args["greenIterationsCount"]
#                 print('MEMORY USAGE: ', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss )
#                 
#                 orthWavefunction = modifiedGramSchmidt_singleOrbital(orbitals,W,m, n, M)
#                 orbitals[:,m] = np.copy(orthWavefunction)
#                 psiIn = np.append( np.copy(orbitals[:,m]), Energies['orbitalEnergies'][m] )
#                 
#                 
#                 ## Update input wavefunctions
#                 if firstInputWavefunction==True:
#                     inputWavefunctions[:,0] = np.copy(psiIn) # fill first column of inputWavefunctions
#                     firstInputWavefunction=False
#                 else:
#                     if (greenIterationsCount-1-mixingStart)<mixingHistoryCutoff:
#                         inputWavefunctions = np.concatenate( ( inputWavefunctions, np.reshape(np.copy(psiIn), (psiIn.size,1)) ), axis=1)
#                         print('Concatenated inputWavefunction.  Now has shape: ', np.shape(inputWavefunctions))
# 
#                     else:
#                         print('Beyond mixingHistoryCutoff.  Replacing column ', (greenIterationsCount-1-mixingStart)%mixingHistoryCutoff)
#                         inputWavefunctions[:,(greenIterationsCount-1-mixingStart)%mixingHistoryCutoff] = np.copy(psiIn)
# 
#                 ## Perform one step of iterations
#                 oldEigenvalue = np.copy(Energies['orbitalEnergies'][m])
# #                 print('Before GI Energies:[orbitalEnergies] ', Energies['orbitalEnergies'])
# #                 greensIteration_FixedPoint, gi_args = greensIteration_FixedPoint_Closure(gi_args)
# #                 r = greensIteration_FixedPoint(psiIn,gi_args)
#                 eigOne_FixedPoint, gi_args = eigenvalueOne_FixedPoint_Closure(gi_args)
#                 r = eigOne_FixedPoint(psiIn, gi_args)
# #                 print('After GI Energies:[orbitalEnergies] ', Energies['orbitalEnergies'])
#                 newEigenvalue = np.copy(Energies['orbitalEnergies'][m])
#                 psiOut = np.append( gi_args["orbitals"][:,m], Energies['orbitalEnergies'][m])
#                 clenshawCurtisNorm = clenshawCurtisNormClosure(W)
#                 errorNorm = clenshawCurtisNorm(psiOut-psiIn)
#                 print('Error Norm: ', errorNorm)
#                 if errorNorm < intraScfTolerance:
#                     Done=True
#                 eigenvalueDiff = np.abs(oldEigenvalue-newEigenvalue)
#                 print('Eigenvalue Diff: ', eigenvalueDiff)
# #                 if ( (eigenvalueDiff < intraScfTolerance) and (gi_args['greenIterationsCount'] > 20) ):  # must have tried to converge wavefunction. If after 20 iteration, allow eigenvalue tolerance to be enough. 
# #                     print('Ending iteration because eigenvalue is converged.')
# #                     Done=True
#                 
#                 
#                 
#                 ## Update output wavefunctions
#                 
#                 if firstOutputWavefunction==True:
# #                     temp = np.append( orbitals[:,m], Energies['orbitalEnergies'][m])
#                     outputWavefunctions[:,0] = np.copy(psiOut) # fill first column of outputWavefunctions
#                     firstOutputWavefunction=False
#                 else:
#                     if (greenIterationsCount-1-mixingStart)<mixingHistoryCutoff:
# #                         temp = np.append( orbitals[:,m], Energies['orbitalEnergies'][m])
#                         outputWavefunctions = np.concatenate( ( outputWavefunctions, np.reshape(np.copy(psiOut), (psiOut.size,1)) ), axis=1)
#                         print('Concatenated outputWavefunction.  Now has shape: ', np.shape(outputWavefunctions))
#                     else:
#                         print('Beyond mixingHistoryCutoff.  Replacing column ', (greenIterationsCount-1-mixingStart)%mixingHistoryCutoff)
# #                         temp = np.append( orbitals[:,m], Energies['orbitalEnergies'][m])
#                         outputWavefunctions[:,(greenIterationsCount-1-mixingStart)%mixingHistoryCutoff] = np.copy(psiOut)
#                 
#                 
#                 ## Compute next input wavefunctions
#                 print('Anderson mixing on the orbital.')
#                 GImixingParameter=0.5
#                 andersonOrbital, andersonWeights = densityMixing.computeNewDensity(inputWavefunctions, outputWavefunctions, GImixingParameter,np.append(W,1.0), returnWeights=True)
#                 Energies['orbitalEnergies'][m] = andersonOrbital[-1]
#                 orbitals[:,m] = andersonOrbital[:-1]
                
                


             
#             print('Used %i iterations for wavefunction %i' %(greenIterationsCount,m))
            print('Used %i iterations for wavefunction %i' %(gi_args["greenIterationsCount"],m))

        
        ## Sort by eigenvalue
        orbitals, Energies['orbitalEnergies'] = sortLargestFirst(orbitals,Energies['orbitalEnergies'])

    #         if vtkExport != False:
    #             filename = vtkExport + '/mesh%03d'%(SCFcount-1) + '.vtk'
    #             Energies['Etotal']xportGridpoints(filename)
    
        printEachIteration=True
    
        if printEachIteration==True:
            header = ['Iteration', 'targetEpsilon', 'orbitalEnergies']
        
            myData = [SCFcount, targetEpsilon-gaugeShift, Energies['orbitalEnergies']]
            
        
            if not os.path.isfile(SCFiterationOutFile):
                myFile = open(SCFiterationOutFile, 'a')
                with myFile:
                    writer = csv.writer(myFile)
                    writer.writerow(header) 
                
            
            myFile = open(SCFiterationOutFile, 'a')
            with myFile:
                writer = csv.writer(myFile)
                writer.writerow(myData)
        
        
        ## Write the restart files
        
                
        
#         if plotSliceOfDensity==True:
#     #             densitySliceSavefile = densityPlotsDir+'/iteration'+str(SCFcount)
#             r, rho = tree.interpolateDensity(xi,yi,zi,xf,yf,zf, numpts, plot=False, save=False)
#         
#     #
#             densities = np.load(densitySliceSavefile+'.npy')
#             densities = np.concatenate( (densities, np.reshape(rho, (numpts,1))), axis=1)
#             np.save(densitySliceSavefile,densities)
            
            
        ## Pack up scf_args
        scf_args['outputDensities']=outputDensities
        scf_args['inputDensities']=inputDensities
        scf_args['SCFcount']=SCFcount
        scf_args['Energies']=Energies
        scf_args['Times']=Times
        scf_args['orbitals']=orbitals
        scf_args['oldOrbitals']=oldOrbitals
        scf_args['Veff']=Veff
    
    
        return
    return eigOneDriverFixedPoint, scf_args



