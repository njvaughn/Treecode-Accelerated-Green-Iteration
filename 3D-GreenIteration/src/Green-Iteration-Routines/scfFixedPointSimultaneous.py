'''
scfFixedPointSimultaneous.py

Performs Green iteration update to each wavefunction before orthogonalizing the set.
Typically this is very wasteful, as the higher energy states don't converge 
until the lower energy states have converged tightly.
This approach is less efficient than the sequential version.
'''

import numpy as np
import os
import csv
import time
import resource

from scipy.optimize import root as scipyRoot
from scipy.optimize.nonlin import BroydenFirst, KrylovJacobian
from scipy.optimize.nonlin import InverseJacobian
from scipy.optimize import broyden1, anderson, brentq
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size() 
from mpiUtilities import global_dot, rprint


from meshUtilities import interpolateBetweenTwoMeshes
import interpolation_wrapper
from fermiDiracDistribution import computeOccupations
import densityMixingSchemes as densityMixing
import treecodeWrappers_distributed as treecodeWrappers
# from orthogonalizationRoutines import modifiedGramSchmidt_singleOrbital_transpose as mgs
from orthogonalizationRoutines import modifiedGramSchmidt as mgs
from greenIterationFixedPoint import greensIteration_FixedPoint_Closure



 
Temperature = 500
KB = 1/315774.6
Sigma = Temperature*KB
def fermiObjectiveFunctionClosure(Energies,nElectrons):
    def fermiObjectiveFunction(fermiEnergy):
                exponentialArg = (Energies['orbitalEnergies']-fermiEnergy)/Sigma
                temp = 1/(1+np.exp( exponentialArg ) )
                return nElectrons - 2 * np.sum(temp)
    return fermiObjectiveFunction

def clenshawCurtisNormClosure_noAppendedEigenvalueWeight(W):
    def clenshawCurtisNorm(psi):
        norm = np.sqrt( global_dot( psi, psi*W, comm ) )
        return norm
    return clenshawCurtisNorm

def clenshawCurtisNormClosure(W):
    def clenshawCurtisNorm(psi):
        appendedWeights = np.append(W, 1.0)   # NOTE: The appended weight was previously set to 10, giving extra weight to the eigenvalue 
        norm = np.sqrt( global_dot( psi, psi*appendedWeights, comm ) )
        return norm
    return clenshawCurtisNorm

def clenshawCurtisNormClosureWithoutEigenvalue(W):
    def clenshawCurtisNormWithoutEigenvalue(psi):
        appendedWeights = np.append(W, 0.0)
        norm = np.sqrt( global_dot( psi, psi*appendedWeights, comm ) )
#         norm = np.sqrt( np.sum( psi*psi*appendedWeights ) )
#         norm = np.sqrt( np.sum( psi[-1]*psi[-1]*appendedWeights[-1] ) )
        return norm
    return clenshawCurtisNormWithoutEigenvalue
    
def sortByEigenvalue(orbitals,orbitalEnergies,verbosity=0):
    newOrder = np.argsort(orbitalEnergies)
    oldEnergies = np.copy(orbitalEnergies)
    for m in range(len(orbitalEnergies)):
        orbitalEnergies[m] = oldEnergies[newOrder[m]]
    if verbosity>1: rprint(rank,'Sorted eigenvalues: ', orbitalEnergies)
    if verbosity>1: rprint(rank,'New order: ', newOrder)
    
    newOrbitals = np.zeros_like(orbitals)
    for m in range(len(orbitalEnergies)):
        newOrbitals[m,:] = orbitals[newOrder[m],:]            
   
    return newOrbitals, orbitalEnergies
      
def scfFixedPointClosureSimultaneous(scf_args): 
    
    def scfFixedPointSimultaneous(RHO,scf_args, abortAfterInitialHartree=False):
        
        verbosity=1
        
        ## Unpack scf_args
        inputDensities = scf_args['inputDensities']
        outputDensities=scf_args['outputDensities']
        SCFcount = scf_args['SCFcount']
        coreRepresentation = scf_args['coreRepresentation']
        nPoints = scf_args['nPoints']
        nOrbitals=scf_args['nOrbitals']
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
        Vext_local=scf_args['Vext_local']
        Vext_local_fine=scf_args['Vext_local_fine']
        gaugeShift=scf_args['gaugeShift']
        orbitals=scf_args['orbitals']
        oldOrbitals=scf_args['oldOrbitals']
        Times=scf_args['Times']
        singularityHandling=scf_args['singularityHandling']
        approximationName=scf_args['approximationName']
        X = scf_args['X']
        Y = scf_args['Y']
        Z = scf_args['Z']
        W = scf_args['W']
        Xf = scf_args['Xf']
        Yf = scf_args['Yf']
        Zf = scf_args['Zf']
        Wf = scf_args['Wf']
        pointsPerCell_coarse = scf_args['pointsPerCell_coarse']
        pointsPerCell_fine = scf_args['pointsPerCell_fine']
        gradientFree = scf_args['gradientFree']
        residuals = scf_args['residuals']
        greenIterationOutFile = scf_args['greenIterationOutFile']
        referenceEigenvalues = scf_args['referenceEigenvalues']
        symmetricIteration=scf_args['symmetricIteration']
        initialGItolerance=scf_args['initialGItolerance']
        finalGItolerance=scf_args['finalGItolerance']
        gradualSteps=scf_args['gradualSteps']
        referenceEnergies=scf_args['referenceEnergies']
        SCFiterationOutFile=scf_args['SCFiterationOutFile']
        wavefunctionFile=scf_args['wavefunctionFile']
        densityFile=scf_args['densityFile']
        outputDensityFile=scf_args['outputDensityFile']
        inputDensityFile=scf_args['inputDensityFile']
        vHartreeFile=scf_args['vHartreeFile']
        auxiliaryFile=scf_args['auxiliaryFile']
        atoms=scf_args['atoms']
        nearbyAtoms=scf_args['nearbyAtoms']
        order=scf_args['order']
        fine_order=scf_args['fine_order']
        regularize=scf_args['regularize']
        epsilon=scf_args['epsilon']
        TwoMeshStart=scf_args['TwoMeshStart']
        
        GItolerances = np.logspace(np.log10(initialGItolerance),np.log10(finalGItolerance),gradualSteps)
#         scf_args['GItolerancesIdx']=0
        
        scf_args['currentGItolerance']=GItolerances[scf_args['GItolerancesIdx']]
        rprint(rank,"Current GI toelrance: ", scf_args['currentGItolerance'])
        
        GImixingHistoryCutoff = 10
         
        SCFcount += 1
        rprint(rank,'\nSCF Count ', SCFcount)
        rprint(rank,'Orbital Energies: ', Energies['orbitalEnergies'])
#         TwoMeshStart=1
        SCFindex = SCFcount
        if SCFcount>TwoMeshStart:
            SCFindex = SCFcount - TwoMeshStart
            
        if SCFcount==1:
            ## For the greedy approach, let the density start as the sum of wavefunctions.
            orbitals, Energies['orbitalEnergies'] = sortByEigenvalue(orbitals,Energies['orbitalEnergies'])
            fermiObjectiveFunction = fermiObjectiveFunctionClosure(Energies,nElectrons)        
            eF = brentq(fermiObjectiveFunction, Energies['orbitalEnergies'][0], 1, xtol=1e-14)
            rprint(rank,'Fermi energy: %f'%eF)
            exponentialArg = (Energies['orbitalEnergies']-eF)/Sigma
            occupations = 2*1/(1+np.exp( exponentialArg ) )
        
        
        if SCFcount>1:
            
            
    
            if (SCFindex-1)<mixingHistoryCutoff:
#             if (len(inputDensities[0,:]))<mixingHistoryCutoff:
                inputDensities = np.concatenate( (inputDensities, np.reshape(RHO, (nPoints,1))), axis=1)
            else:
                inputDensities[:,(SCFindex-1)%mixingHistoryCutoff] = np.copy(RHO)
        
            if SCFcount == TwoMeshStart:
                inputDensities = np.zeros((nPoints,1))         
                inputDensities[:,0] = np.copy(RHO)
                
                outputDensities = np.zeros((nPoints,1))

     
        
    
        ### Compute Veff
        """ 
        Compute new electron-electron potential and update pointwise potential values 
        """
        
        # interpolate density to fine mesh for computing hartree potential
#         print(pointsPerCell_coarse)
#         print(pointsPerCell_fine)
#         print(len(X))
#         print(len(Xf))
#         exit(-1)
        if len(X) != len(Xf):
            print("Interpolating density from %i to %i point mesh." %(len(X),len(Xf)))
#             start=time.time()
#             RHOf = interpolateBetweenTwoMeshes(X, Y, Z, RHO, pointsPerCell_coarse,
#                                                    Xf, Yf, Zf, pointsPerCell_fine)
#             end=time.time()
#             print("Original interpolation time: ", end-start)
#             
            numberOfCells=len(pointsPerCell_coarse)
#             start=time.time()
#             print("pointsPerCell_coarse = ", pointsPerCell_coarse[0:5])
#             print("pointsPerCell_fine = ", pointsPerCell_fine[0:5])
#             if numberOfCells
#             print("Type: ", pointsPerCell_fine.dtype)
            RHOf = interpolation_wrapper.callInterpolator(X,  Y,  Z,  RHO, pointsPerCell_coarse,
                                                           Xf, Yf, Zf, pointsPerCell_fine, 
                                                           numberOfCells, order)
            
            
            
            
#             end=time.time()
#             comm.barrier() 
#             print("External interpolation time: ", end-start)
#             print("Difference: ", np.max( np.abs(RHOf-RHOf2)))
#             
#             fCount=0
#             f2Count=0
#             for i in range(len(RHOf)):
#                 
#                 if abs(RHOf[i])==0.0:
#                     fCount+=1
#                 if abs(RHOf2[i])==0.0:
#                     f2Count+=1
# #                 if abs(RHOf[i]-RHOf2[i])>1e-12:
# #                     print(i,RHOf[i], RHOf2[i])
#                     
#             print("Number of zeros: ", fCount, f2Count)
#             exit(-1)
#             print("RHOf = ", RHOf)
#             print("RHOf2 = ", RHOf2)
#             
#             
#             start=0
#             for i in range(numberOfCells):
#                 if pointsPerCell_fine[i]==512:
#                     for j in range(512):
#                         print(RHOf[start+j], RHOf2[start+j])
#                     exit(-1)
#                 start+=int(pointsPerCell_fine[i])
#             exit(-1)
        else:
#             print("WHY IS LEN(X)=LEN(Xf)?")
#             exit(-1)
            RHOf=RHO
#         if order!=fine_order:
#             RHOf = interpolateBetweenTwoMeshes_variableOrder(X, Y, Z, RHO, order,
#                                                    Xf, Yf, Zf, fine_order) 
#         else:
#             RHOf=RHO
        
        
        if treecode==False:
            V_hartreeNew = np.zeros(nPoints)
            densityInput = np.transpose( np.array([X,Y,Z,RHO,W]) )
            V_hartreeNew = cpuHartreeGaussianSingularitySubract(densityInput,densityInput,V_hartreeNew,gaussianAlpha*gaussianAlpha)
            Times['timePerConvolution'] = time.time()-start
            rprint(rank,'Convolution time: ', time.time()-start)
        else:    
            if singularityHandling=='skipping':
                if regularize==False:
                    rprint(rank,"Using singularity skipping in Hartree solve.")
                    kernelName = "coulomb"
                    numberOfKernelParameters=1
                    kernelParameters=np.array([0.0])
                elif regularize==True:
                    rprint(rank,"Using regularize coulomb kernel with epsilon = ", epsilon)
                    kernelName = "regularized-coulomb"
                    numberOfKernelParameters=1
                    kernelParameters=np.array([epsilon])
                else:
                    print("What should regularize be in SCF?")
                    exit(-1)
            elif singularityHandling=='subtraction':
                if regularize==False:                    
                    rprint(rank,"Using singularity subtraction in Hartree solve.")
                    kernelName = "coulomb"
                    numberOfKernelParameters=1
                    kernelParameters=np.array([gaussianAlpha])
                elif regularize==True:
                    rprint(rank,"Using SS and regularization for Hartree solve.")
                    kernelName="regularized-coulomb"
                    numberOfKernelParameters=2
                    kernelParameters=np.array([gaussianAlpha,epsilon])
                    
            else: 
                rprint(rank,"What should singularityHandling be?")
                return
            start = MPI.Wtime()
            
            
#             print("Rank %i calling treecode through wrapper..." %(rank))
            
            treecode_verbosity=0
#             singularityHandling="skipping"
#             print("Forcing the Hartree solve to use singularity skipping.")

            rprint(rank,"Performing Hartree solve on %i mesh points" %(len(Xf)))
            rprint(rank,"Coarse order ", order)
            rprint(rank,"Fine order   ", fine_order)
            V_hartreeNew = treecodeWrappers.callTreedriver(len(X), len(Xf), 
                                                           np.copy(X), np.copy(Y), np.copy(Z), np.copy(RHO), 
                                                           np.copy(Xf), np.copy(Yf), np.copy(Zf), np.copy(RHOf), np.copy(Wf),
                                                           kernelName, numberOfKernelParameters, kernelParameters, singularityHandling, approximationName,
                                                           treecodeOrder, theta, maxParNode, batchSize, GPUpresent, treecode_verbosity)
            
#             V_hartreeNew = treecodeWrappers.callTreedriver(len(X), len(Xf), 
#                                                            X, Y, Z, RHO, 
#                                                            Xf, Yf, Zf, RHOf, Wf,
#                                                            kernelName, numberOfKernelParameters, kernelParameters, singularityHandling, approximationName,
#                                                            treecodeOrder, theta, maxParNode, batchSize, GPUpresent, treecode_verbosity)
#              
            
#             V_hartreeNew = treecodeWrappers.callTreedriver(len(X), len(X), 
#                                                            np.copy(X), np.copy(Y), np.copy(Z), np.copy(RHO), 
#                                                            np.copy(X), np.copy(Y), np.copy(Z), np.copy(RHO), np.copy(W),
#                                                            kernelName, numberOfKernelParameters, kernelParameters, singularityHandling, approximationName,
#                                                            treecodeOrder, theta, maxParNode, batchSize, GPUpresent, verbosity)
            
#             V_hartreeNew += 2.0*np.pi*gaussianAlpha*gaussianAlpha*RHO
            
            
            Times['timePerConvolution'] = MPI.Wtime()-start
            rprint(rank,'Convolution time: ', MPI.Wtime()-start)
            
#             # interpolate back to coarse mesh
#             V_hartreeNew = interpolateBetweenTwoMeshes(Xf, Yf, Zf, V_hartreeNewf, fine_order,
#                                                X, Y, Z, order) 
        
      
        
        
        """ 
        Compute the new orbital and total energies 
        """
        
        ## Energy update after computing Vhartree
        
        comm.barrier()    
#         Energies['Ehartree'] = 1/2*np.sum(W * RHO * V_hartreeNew)
#         VhartreeNorm = np.sqrt( global_dot(W,V_hartreeNew*V_hartreeNew, comm) )
#         rprint(rank,"VHartreeNew norm = ", VhartreeNorm)
        Energies['Ehartree'] = 1/2*global_dot(W, RHO * V_hartreeNew, comm)
        FIRST_SCF_ELECTROSTATICS_DFTFE=-4.0049522077687829e+00
        if abortAfterInitialHartree==True:
            Energies["Repulsion"] = global_dot(RHO, Vext_local*W, comm)
        
            Energies['totalElectrostatic'] = Energies["Ehartree"] + Energies["Enuclear"] + Energies["Repulsion"]
            rprint(rank,"Energies['Ehartree'] after initial convolution: ", Energies['Ehartree'])
#             rprint(rank,"Electrostatics error after initial convolution: ", Energies['totalElectrostatic']-referenceEnergies["Eelectrostatic"])
            rprint(rank,"Electrostatics error after initial convolution: ", Energies['totalElectrostatic']-FIRST_SCF_ELECTROSTATICS_DFTFE)
            return np.zeros(nPoints)
        
        
    
        exchangeOutput = exchangeFunctional.compute(RHO)
        correlationOutput = correlationFunctional.compute(RHO)
#         Energies['Ex'] = np.sum( W * RHO * np.reshape(exchangeOutput['zk'],np.shape(RHO)) )
#         Energies['Ec'] = np.sum( W * RHO * np.reshape(correlationOutput['zk'],np.shape(RHO)) )
        
        Energies['Ex'] = global_dot( W, RHO * np.reshape(exchangeOutput['zk'],np.shape(RHO)), comm )
        Energies['Ec'] = global_dot( W, RHO * np.reshape(correlationOutput['zk'],np.shape(RHO)), comm )
        
        Vx = np.reshape(exchangeOutput['vrho'],np.shape(RHO))
        Vc = np.reshape(correlationOutput['vrho'],np.shape(RHO))
        
#         Energies['Vx'] = np.sum(W * RHO * Vx)
#         Energies['Vc'] = np.sum(W * RHO * Vc)

        Energies['Vx'] = global_dot(W, RHO * Vx,comm)
        Energies['Vc'] = global_dot(W, RHO * Vc,comm)
        
        Veff_local = V_hartreeNew + Vx + Vc + Vext_local + gaugeShift
        
        
        if SCFcount==1: # generate initial guesses for eigenvalues
            Energies['Eold']=-10
#             for m in range(nOrbitals):
#                 Energies['orbitalEnergies'][m]=-1.0
# #             orbitals, Energies['orbitalEnergies'] = sortByEigenvalue(orbitals, Energies['orbitalEnergies'])
# #             for m in range(nOrbitals):
# #                 if Energies['orbitalEnergies'][m] > 0:
# #                     Energies['orbitalEnergies'][m] = -0.5
        
        
        ## Sort by eigenvalue
#         orbitals, Energies['orbitalEnergies'] = sortByEigenvalue(orbitals,Energies['orbitalEnergies'])
           
        
        ## Solve the eigenvalue problem
        if SCFcount>1:
            fermiObjectiveFunction = fermiObjectiveFunctionClosure(Energies,nElectrons)        
            eF = brentq(fermiObjectiveFunction, Energies['orbitalEnergies'][0], 1, xtol=1e-14)
            rprint(rank,'Fermi energy: %f'%eF)
            exponentialArg = (Energies['orbitalEnergies']-eF)/Sigma
            previousOccupations = 2*1/(1+np.exp( exponentialArg ) )
        elif SCFcount==1: 
            previousOccupations = np.ones(nOrbitals)
            
            
        ##############################################################################################
        """ Everything above this line is the same for sequential and simultaneous Green Iteration """
        ##############################################################################################
        
        ## Structure
        
        #1. Construct input vector consisting of all wavefunctions (possible w/ numpy resize): oldOrbitals
        #2. Call Green Iteration on each wavefunction (without orthogonalization)
        #3. Perform a single MGS orthogonalization
        #4. Construct output vector consisting of all updated wavfunctions
        #5. Call the mixing scheme on the whole set
        
        nPointsPlus1=nPoints+1
        
        set_weights=np.zeros(nOrbitals*nPoints)
        for m in range(nOrbitals):
            set_weights[m*nPoints:(m+1)*nPoints] = W
            
#         print("Sum of set of weights: ", np.sum(set_weights))
        firstInputWavefunction=True
        firstOutputWavefunction=True
        inputWavefunctions = np.zeros((nOrbitals*nPoints,1))
        outputWavefunctions =  np.zeros((nOrbitals*nPoints,1))
#         mixingStart = np.copy( gi_args['greenIterationsCount'] )
        
        if SCFcount==1:
            mixingStart = 3
        elif SCFcount>1:
            mixingStart = 0
        
        eigenvalueResiduals=np.ones_like(residuals)
        updatedOrbitals=np.empty_like(residuals)
        for m in range(nOrbitals):
            updatedOrbitals[m]=True
        iterationCount=0
        converged=False
        while not converged:
            iterationCount+=1 
            AtLeastOneWavefunctionUpdated=False
            
            set_inputVectors = np.resize(oldOrbitals, (nOrbitals*nPoints,))
            updatedCounter=0
            for m in range(nOrbitals): 
#                 if ( previousOccupations[m] > 1e-12 ):
                if ( (previousOccupations[m] > 1e-12) and (eigenvalueResiduals[m]>(scf_args['currentGItolerance']/30)) ):
#                     rprint(rank,"Eigenvalue residuals: ", eigenvalueResiduals)
                    AtLeastOneWavefunctionUpdated=True
                    if verbosity>1: rprint(rank,'Working on orbital %i' %m)
                    if verbosity>1: rprint(rank,'MEMORY USAGE: %i' %resource.getrusage(resource.RUSAGE_SELF).ru_maxrss )
                    
            #                         
                    updatedCounter+=1          
                    greenIterationsCount=1
    #                 print("Forcing singularityHandling to be skipping for Green's iteration.")
                    gi_args = {'orbitals':orbitals,'oldOrbitals':oldOrbitals, 'Energies':Energies, 'Times':Times, 
                               'Veff_local':Veff_local, 'Vext_local':Vext_local, 'Vext_local_fine':Vext_local_fine,
                                   'symmetricIteration':symmetricIteration,'GPUpresent':GPUpresent,
                                   'singularityHandling':singularityHandling, 'approximationName':approximationName,
                                   'treecode':treecode,'treecodeOrder':treecodeOrder,'theta':theta, 'maxParNode':maxParNode,'batchSize':batchSize,
                                   'nPoints':nPoints, 'm':m, 'X':X,'Y':Y,'Z':Z,'W':W,'Xf':Xf,'Yf':Yf,'Zf':Zf,'Wf':Wf,'gradientFree':gradientFree,
                                   'SCFcount':SCFcount,'greenIterationsCount':greenIterationsCount,
                                   'residuals':residuals,
                                   'eigenvalueResiduals':eigenvalueResiduals,
                                   'greenIterationOutFile':greenIterationOutFile,
                                   'referenceEigenvalues':referenceEigenvalues,
                                   'updateEigenvalue':True,
                                   'coreRepresentation':coreRepresentation,
                                   'atoms':atoms,
                                   'nearbyAtoms':nearbyAtoms,
                                   'order':order,
                                   'fine_order':fine_order,
                                   'regularize':regularize, 'epsilon':epsilon,
                                   'pointsPerCell_coarse':pointsPerCell_coarse,
                                   'pointsPerCell_fine':pointsPerCell_fine,
                                   'TwoMeshStart':TwoMeshStart,
                                   'singleWavefunctionOrthogonalization':False
                                   } 
                    
                    n,M = np.shape(orbitals)
                    resNorm=1.0 

                    comm.barrier()


                    oldEigenvalue = np.copy(Energies['orbitalEnergies'][m])

                    for calls in range(1):
                        psiIn = np.append( np.copy(orbitals[m,:]), Energies['orbitalEnergies'][m] )
                        greensIteration_FixedPoint, gi_args = greensIteration_FixedPoint_Closure(gi_args)
                        r = greensIteration_FixedPoint(psiIn, gi_args)
                    newEigenvalue = np.copy(Energies['orbitalEnergies'][m])
                    eigenvalueDiff=np.abs(oldEigenvalue - newEigenvalue )
                    comm.barrier()
                    if verbosity>1: rprint(rank,'eigenvalueDiff = %f' %eigenvalueDiff)
                        
     
                    clenshawCurtisNorm = clenshawCurtisNormClosure(W)
                    resNorm = clenshawCurtisNorm(r)
                    
                    if verbosity>1: rprint(rank,'CC norm of residual vector: %f'%resNorm)
#                     if eigenvalueDiff < resNorm/10:
#                         resNorm = eigenvalueDiff
#                         if verbosity>0: rprint(rank,'Using eigenvalueDiff: %f' %resNorm)
                else:
                    
                    if (eigenvalueResiduals[m]<scf_args['currentGItolerance']/30):
                        if verbosity>1: rprint(rank,"Not updating wavefunction %i because its eigenvalue is converged." %m)
                    elif (previousOccupations[m] < 1e-12):
                        if verbosity>1: rprint(rank,"Not updating wavefunction %i because it is unoccupied." %m)
                        
                    orbitals[m] = np.copy(oldOrbitals[m])
                    updatedOrbitals[m]=False
#                     set_weights[m*nPoints:(m+1)*nPoints] = np.zeros(nPoints)

            if AtLeastOneWavefunctionUpdated==False:
                converged=True
            else:
                ## Orthogonalize.
                
                
                
                orbitals = mgs(orbitals, W, comm)
                
                set_outputVectors = np.resize(orbitals, (nOrbitals*nPoints,))
                
                
                ## Measure residuals of individual wavefunctions, just for analyzing
                maxResidual=0.0
                sumResidual=0.0
                maxID=-1 
                for m in range(nOrbitals):
                    clenshawCurtisNorm = clenshawCurtisNormClosure_noAppendedEigenvalueWeight(W)
                    resNorm = clenshawCurtisNorm(set_outputVectors[m*nPoints:(m+1)*nPoints]-set_inputVectors[m*nPoints:(m+1)*nPoints])
                    sumResidual+=resNorm**2
                    if resNorm>maxResidual:
                        maxID=m
                        maxResidual=resNorm
                
                rprint(rank,"\n============================================")
                rprint(rank,"============================================")
                rprint(rank,"Iteration %i, updated %i of %i wavefunctions." %(iterationCount,updatedCounter,nOrbitals))
                clenshawCurtisNorm = clenshawCurtisNormClosure_noAppendedEigenvalueWeight(set_weights)
                resNorm = clenshawCurtisNorm(set_outputVectors-set_inputVectors)
                rprint(rank,"Residual norm = %1.3e" %resNorm)
                rprint(rank,"Largest individual: wavefunction %i with residual %1.3e" %(maxID,maxResidual))
                rprint(rank,"Sum of individual norms = %1.3e" %np.sqrt(sumResidual) )
                rprint(rank,"============================================")
                rprint(rank,"============================================\n")
                
                
                ## Sort if needed...
                if ( (iterationCount==mixingStart) and (SCFcount==1) ):
                    oldEnergies=np.copy(Energies['orbitalEnergies'])
                    if verbosity>1: rprint("Eigenvalues before sorting: ", oldEnergies)
                    orbitals, Energies['orbitalEnergies'] = sortByEigenvalue(orbitals,Energies['orbitalEnergies'])
                    oldOrbitals, oldEnergies = sortByEigenvalue(oldOrbitals,oldEnergies)
                    
                    set_inputVectors = np.resize(oldOrbitals, (nOrbitals*nPoints,))
                    set_outputVectors = np.resize(orbitals, (nOrbitals*nPoints,))
                    
                    oldOrbitals=np.copy(orbitals)
#                     input()
                    
                if resNorm<scf_args['currentGItolerance']:
                    converged=True
                else:
                    oldOrbitals=np.copy(orbitals)
                    
                    
                ## Perform Mixing
                
                    
                # Update input vectors history
                if iterationCount>=mixingStart:
                    if firstInputWavefunction==True:
                        inputWavefunctions[:,0] = np.copy(set_inputVectors) # fill first column of inputWavefunctions
                        firstInputWavefunction=False
                    else:
                        if (iterationCount-1-mixingStart)<GImixingHistoryCutoff:
                            inputWavefunctions = np.concatenate( ( inputWavefunctions, np.reshape(np.copy(set_inputVectors), (nPoints*nOrbitals,1)) ), axis=1)
                            if verbosity>1: rprint(rank,'Concatenated inputWavefunction.  Now has shape: ', np.shape(inputWavefunctions))
        
                        else:
                            if verbosity>1: rprint(rank,'Beyond GImixingHistoryCutoff.  Replacing column ', (iterationCount-1-mixingStart)%GImixingHistoryCutoff)
                            inputWavefunctions[:,(iterationCount-1-mixingStart)%GImixingHistoryCutoff] = np.copy(set_inputVectors)
                    
                    # Update output vectors history
                    if firstOutputWavefunction==True:
        #                     temp = np.append( orbitals[m,:], Energies['orbitalEnergies'][m])
                        outputWavefunctions[:,0] = np.copy(set_outputVectors) # fill first column of outputWavefunctions
                        firstOutputWavefunction=False
                    else:
                        if (iterationCount-1-mixingStart)<GImixingHistoryCutoff:
        #                         temp = np.append( orbitals[m,:], Energies['orbitalEnergies'][m])
                            outputWavefunctions = np.concatenate( ( outputWavefunctions, np.reshape(np.copy(set_outputVectors), (nOrbitals*nPoints,1)) ), axis=1)
                            if verbosity>1: rprint(rank,'Concatenated outputWavefunction.  Now has shape: ', np.shape(outputWavefunctions))
                        else:
                            if verbosity>1: rprint(rank,'Beyond GImixingHistoryCutoff.  Replacing column ', (iterationCount-1-mixingStart)%GImixingHistoryCutoff)
        #                         temp = np.append( orbitals[m,:], Energies['orbitalEnergies'][m])
                            outputWavefunctions[:,(iterationCount-1-mixingStart)%GImixingHistoryCutoff] = np.copy(set_outputVectors)
                    
                    
                    
                    
                    ## Compute next input wavefunctions
                    if verbosity>1: rprint(rank,'Anderson mixing on the wavefunctions.')
                    GImixingParameter=0.25
                    
                    ## Mix all wavefunctions simultaneously
                    andersonOrbitals, andersonWeights = densityMixing.computeNewDensity(inputWavefunctions, outputWavefunctions, GImixingParameter,set_weights, returnWeights=True)
                    orbitals = np.reshape(andersonOrbitals,(nOrbitals,nPoints))
                
#                     ## Mix each wavefunction individually
#                     for m in range(nOrbitals):
#                         if updatedOrbitals[m]==True:
#     #                         rprint(rank,"Anderson mixing on orbital %i" %m)
#                             andersonOrbital, andersonWeights = densityMixing.computeNewDensity(inputWavefunctions[m*nPoints:(m+1)*nPoints,:], outputWavefunctions[m*nPoints:(m+1)*nPoints,:], GImixingParameter,W, returnWeights=True)
#             #             Energies['orbitalEnergies'][m] = andersonOrbital[-1]
#                             orbitals[m,:] = andersonOrbital
    #    
                    
                    
                    ## Periodically update the occupations, might be working on an unoccupied state
                    if iterationCount%4==0:
                        fermiObjectiveFunction = fermiObjectiveFunctionClosure(Energies,nElectrons)        
                        eF = brentq(fermiObjectiveFunction, Energies['orbitalEnergies'][0], 1, xtol=1e-14)
                        if verbosity>1: rprint(rank,'Fermi energy: ', eF)
                        exponentialArg = (Energies['orbitalEnergies']-eF)/Sigma
                        occupations = 2*1/(1+np.exp( exponentialArg ) )  # these are # of electrons, not fractional occupancy.  Hence the 2*
                        if verbosity>1: rprint(rank, "New occupations: ", occupations)
                        previousOccupations=occupations

        
        orbitals, Energies['orbitalEnergies'] = sortByEigenvalue(orbitals,Energies['orbitalEnergies'])
        
        fermiObjectiveFunction = fermiObjectiveFunctionClosure(Energies,nElectrons)        
        eF = brentq(fermiObjectiveFunction, Energies['orbitalEnergies'][0], 1, xtol=1e-14)
        if verbosity>0: rprint(rank,'Fermi energy: ', eF)
        exponentialArg = (Energies['orbitalEnergies']-eF)/Sigma
        occupations = 2*1/(1+np.exp( exponentialArg ) )  # these are # of electrons, not fractional occupancy.  Hence the 2*
    
    #         occupations = computeOccupations(Energies['orbitalEnergies'], nElectrons, Temperature)
        if verbosity>0: rprint(rank,'Occupations: ', occupations)
        Energies['Eband'] = np.sum( (Energies['orbitalEnergies']-Energies['gaugeShift']) * occupations)
    
     
    
        
    
        
    
        oldDensity = np.copy(RHO)
        
        RHO = np.zeros(nPoints)
        for m in range(nOrbitals):
            RHO += orbitals[m,:]**2 * occupations[m]
        newDensity = np.copy(RHO)
        
        if verbosity>0: rprint(rank,"Integral of old RHO ", global_dot( oldDensity,W,comm ) )
        if verbosity>0: rprint("Integral of new RHO ", global_dot( newDensity,W,comm ) )
        
        if verbosity>0: rprint(rank,"NORMALIZING NEW RHO")
        densityIntegral=global_dot( newDensity,W,comm )
        newDensity *= nElectrons/densityIntegral
        
    
        if SCFcount==1: 
            outputDensities[:,0] = np.copy(newDensity)
        else:
            
    #             outputDensities = np.concatenate( ( outputDensities, np.reshape(np.copy(newDensity), (nPoints,1)) ), axis=1)
            
#             if (SCFcount-1)<mixingHistoryCutoff:
#             if (len(outputDensities[0,:]))<mixingHistoryCutoff:
            if (SCFindex-1)<mixingHistoryCutoff:
                outputDensities = np.concatenate( (outputDensities, np.reshape(np.copy(newDensity), (nPoints,1))), axis=1)
#                 print('Concatenated outputDensity.  Now has shape: ', np.shape(outputDensities))
            else:
#                 print('Beyond mixingHistoryCutoff.  Replacing column ', (SCFcount-1)%mixingHistoryCutoff)
    #                                 print('Shape of oldOrbitals[m,:]: ', np.shape(oldOrbitals[m,:]))
                outputDensities[:,(SCFindex-1)%mixingHistoryCutoff] = newDensity
            
            if SCFcount == TwoMeshStart:
                outputDensities = np.zeros((nPoints,1))         
                outputDensities[:,0] = np.copy(newDensity)
    
         
#         if SCFcount == TwoMeshStart:
#              outputDensities[:,0] = np.copy(newDensity)
            
        if verbosity>0: rprint(rank,'outputDensities[0,:] = ', outputDensities[0,:])
        
        rprint(rank,"\n\n\n\n\nShape of input densities: ", np.shape(inputDensities))
        rprint(rank,"Shape of output densities: ", np.shape(outputDensities))
        rprint(rank,"\n\n\n\n\n")
#         input("Press Enter to continue...")
#         print('outputDensities[:,0:3] = ', outputDensities[:,0:3])
        
    #         print('Sample of output densities:')
    #         print(outputDensities[0,:])    
#         integratedDensity = np.sum( newDensity*W )
        integratedDensity = global_dot( newDensity, W, comm )
#         densityResidual = np.sqrt( np.sum( (newDensity-oldDensity)**2*W ) )
        densityResidual = np.sqrt( global_dot( (newDensity-oldDensity)**2,W,comm ) )
        if verbosity>0: rprint(rank,'Integrated density: ', integratedDensity)
        rprint(rank,'Density Residual ', densityResidual)
        
        
        Energies["Repulsion"] = global_dot(newDensity, Vext_local*W, comm)
        
        Energies['Eband'] = np.sum( (Energies['orbitalEnergies']-Energies['gaugeShift']) * occupations)
        Energies['Etotal'] = Energies['Eband'] - Energies['Ehartree'] + Energies['Ex'] + Energies['Ec'] - Energies['Vx'] - Energies['Vc'] + Energies['Enuclear']
        Energies['totalElectrostatic'] = Energies["Ehartree"] + Energies["Enuclear"] + Energies["Repulsion"]
        
        ## This might not be needed, because Eext is already captured in the band energy, which includes both local and nonlocal
#         if coreRepresentation=="Pseudopotential":
#             Eext_nl=0.0
#             for m in range(nOrbitals):
#                 Vext_nl = np.zeros(nPoints)
#                 for atom in atoms:
#                     Vext_nl += atom.V_nonlocal_pseudopotential_times_psi(X,Y,Z,orbitals[m,:],W,comm)
#                 Eext_nl += global_dot(orbitals[m,:], Vext_nl,comm)
#             Energies['Etotal'] += Eext_nl
    
        for m in range(nOrbitals):
            if verbosity>0: rprint(rank,'Orbital %i error: %1.3e' %(m, Energies['orbitalEnergies'][m]-referenceEigenvalues[m]-Energies['gaugeShift']))
        
        
        energyResidual = abs( Energies['Etotal'] - Energies['Eold'] )  # Compute the energyResidual for determining convergence
#         energyError = abs( Energies['Etotal'] - Energies['Eold'] )  # Compute the energyResidual for determining convergence
        Energies['Eold'] = np.copy(Energies['Etotal'])
        
        
        
        """
        Print results from current iteration
        """
    
        rprint(rank,'Orbital Energies: ', Energies['orbitalEnergies']-Energies['gaugeShift']) 
    
        rprint(rank,'Updated V_x:                           %.10f Hartree' %Energies['Vx'])
        rprint(rank,'Updated V_c:                           %.10f Hartree' %Energies['Vc'])
        
        rprint(rank,'Updated Band Energy:                   %.10f H, %.10e H' %(Energies['Eband'], Energies['Eband']-referenceEnergies['Eband']) )
    #         print('Updated Kinetic Energy:                 %.10f H, %.10e H' %(Energies['kinetic'], Energies['kinetic']-Ekinetic) )
        rprint(rank,'Updated E_Hartree:                      %.10f H, %.10e H' %(Energies['Ehartree'], Energies['Ehartree']-referenceEnergies['Ehartree']) )
        rprint(rank,'Updated E_x:                           %.10f H, %.10e H' %(Energies['Ex'], Energies['Ex']-referenceEnergies['Eexchange']) )
        rprint(rank,'Updated E_c:                           %.10f H, %.10e H' %(Energies['Ec'], Energies['Ec']-referenceEnergies['Ecorrelation']) )
        rprint(rank,'Updated totalElectrostatic:            %.10f H, %.10e H' %(Energies['totalElectrostatic'], Energies['totalElectrostatic']-referenceEnergies["Eelectrostatic"]))
        rprint(rank,'Electrostatics w.r.t. SCF 1:           %.10f H, %.10e H' %(Energies['totalElectrostatic'], Energies['totalElectrostatic']-FIRST_SCF_ELECTROSTATICS_DFTFE))
        rprint(rank,'Total Energy:                          %.10f H, %.10e H' %(Energies['Etotal'], Energies['Etotal']-referenceEnergies['Etotal']))
        rprint(rank,'Energy Residual:                        %.3e' %energyResidual)
        rprint(rank,'Density Residual:                       %.3e\n\n'%densityResidual)
    
        scf_args['energyResidual']=energyResidual
        scf_args['densityResidual']=densityResidual
        
        
            
    #         if vtkExport != False:
    #             filename = vtkExport + '/mesh%03d'%(SCFcount-1) + '.vtk'
    #             Energies['Etotal']xportGridpoints(filename)
    
        printEachIteration=True
    
        if printEachIteration==True:
            header = ['Iteration', 'densityResidual', 'orbitalEnergies','bandEnergy', 'kineticEnergy', 
                      'exchangeEnergy', 'correlationEnergy', 'hartreeEnergy', 'totalEnergy', 'GItolerance']
         
            myData = [SCFcount, densityResidual, Energies['orbitalEnergies']-Energies['gaugeShift'], Energies['Eband'], Energies['kinetic'], 
                      Energies['Ex'], Energies['Ec'], Energies['Ehartree'], Energies['Etotal'], scf_args['currentGItolerance']]

#             header = ['Iteration', 'densityResidual', 'orbitalEnergies','bandEnergy', 'kineticEnergy', 
#                       'exchangeEnergy', 'correlationEnergy', 'hartreeEnergy', 'totalEnergy']
#         
#             myData = [SCFcount, densityResidual, Energies['orbitalEnergies'], Energies['Eband'], Energies['kinetic'], 
#                       Energies['Ex'], Energies['Ec'], Energies['Ehartree'], Energies['Etotal']]
            
            if rank==0:
                if not os.path.isfile(SCFiterationOutFile):
                    myFile = open(SCFiterationOutFile, 'a')
                    with myFile:
                        writer = csv.writer(myFile)
                        writer.writerow(header) 
                    
                
                myFile = open(SCFiterationOutFile, 'a')
                with myFile:
                    writer = csv.writer(myFile)
                    writer.writerow(myData)
                
#         if scf_args['currentGItolerance'] > scf_args['finalGItolerance']: # desired final tolerance
        if scf_args['GItolerancesIdx'] < scf_args['gradualSteps']-1: # GItolerancesIdx goes from 0 to gradualSteps-1
            scf_args['GItolerancesIdx']+=1
            scf_args['currentGItolerance'] = GItolerances[scf_args['GItolerancesIdx']]
            rprint(rank,'Reducing GI tolerance to ', scf_args['currentGItolerance'])
        
        
        ## Write the restart files
        
        # save arrays 
        try:
            np.save(wavefunctionFile, orbitals)
            
    #             sources = tree.extractLeavesDensity()
            np.save(densityFile, RHO)
            np.save(outputDensityFile, outputDensities)
            np.save(inputDensityFile, inputDensities)
            
            np.save(vHartreeFile, V_hartreeNew)
            
            
            
            # make and save dictionary
            auxiliaryRestartData = {}
            auxiliaryRestartData['SCFcount'] = SCFcount
            auxiliaryRestartData['totalIterationCount'] = Times['totalIterationCount']
            auxiliaryRestartData['eigenvalues'] = Energies['orbitalEnergies']
            auxiliaryRestartData['Eold'] = Energies['Eold']
    
            np.save(auxiliaryFile, auxiliaryRestartData)
        except FileNotFoundError:
            pass
                
        
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
        scf_args['Veff_local']=Veff_local
    
    
        return newDensity-oldDensity
    return scfFixedPointSimultaneous, scf_args



