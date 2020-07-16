import numpy as np
import os
import csv
import time
import resource

import sys
sys.path.insert(1, '/Users/nathanvaughn/Documents/GitHub/TAGI/3D-GreenIterations/src/utilities')
sys.path.insert(1, '/Users/nathanvaughn/Documents/GitHub/TAGI/3D-GreenIterations/src/dataStructures')
sys.path.insert(1, '/home/njvaughn/TAGI/3D-GreenIterations/src/utilities')
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
import BaryTreeInterface as BT
# from orthogonalizationRoutines import modifiedGramSchmidt_singleOrbital_transpose as mgs
import orthogonalization_wrapper as ORTH
import moveData_wrapper as MOVEDATA
from greenIterationFixedPoint import greensIteration_FixedPoint_Closure


def print_eigs_and_occupations(eigs,occupations,errors):
    rprint(rank," ")
    rprint(rank," "+'-'*60)
    rprint(rank,"| Index\t Eigenvalue \t\t Occupation \t  Error      |")
    rprint(rank,'|'+'-'*60+'|')
    for i in range(len(eigs)):
        rprint(rank,"|%4i:\t %1.10e \t %1.4e \t % 1.4e |" %(i,eigs[i],occupations[i],errors[i]))
    rprint(rank," "+'-'*60+"\n")
    
 
Temperature = 100
KB = 1/315774.6
Sigma = Temperature*KB

def fermiObjectiveFunctionClosure(Energies,nElectrons):
    def fermiObjectiveFunction(fermiEnergy):
        exponentialArg = (Energies['orbitalEnergies']-fermiEnergy)/Sigma
        temp = 1/(1+np.exp( exponentialArg ) )
#         rprint(rank, "input fermiEnergy = ", fermiEnergy)
#         rprint(rank, "Energies['orbitalEnergies'] = ", Energies['orbitalEnergies'])
#         rprint(rank, "exponentialArg = ", exponentialArg)
#         rprint(rank, "temp = ", temp)
        
        return nElectrons - 2 * np.sum(temp)
    return fermiObjectiveFunction


def clenshawCurtisNormClosure(W):
    def clenshawCurtisNorm(psi):
#         appendedWeights = np.append(W, 1.0)   # NOTE: The appended weight was previously set to 10, giving extra weight to the eigenvalue 
        appendedWeights = np.append(np.zeros_like(W), 10.0)   # NOTE: The appended weight was previously set to 10, giving extra weight to the eigenvalue 
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
    if verbosity>0: rprint(rank,'Sorted eigenvalues: ', orbitalEnergies)
    if verbosity>0: rprint(rank,'New order: ', newOrder)
    
    newOrbitals = np.zeros_like(orbitals)
    for m in range(len(orbitalEnergies)):
        newOrbitals[m,:] = orbitals[newOrder[m],:]            
   
    return newOrbitals, orbitalEnergies
      
def scfFixedPointClosure(scf_args): 
    
    def scfFixedPoint(RHO,scf_args, abortAfterInitialHartree=False):
        
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
        CORECHARGERHO=scf_args['CORECHARGERHO']
        
        NLCC_RHO = RHO+CORECHARGERHO
        
        
        GItolerances = np.logspace(np.log10(initialGItolerance),np.log10(finalGItolerance),gradualSteps)
#         scf_args['GItolerancesIdx']=0
        
        scf_args['currentGItolerance']=GItolerances[scf_args['GItolerancesIdx']]
        rprint(rank,"Current GI toelrance: ", scf_args['currentGItolerance'])
        
        GImixingHistoryCutoff = 10
         
        SCFcount += 1
        rprint(rank,'\n')
        rprint(rank,'SCF Count ', SCFcount)
        if verbosity>0: rprint(rank,'Orbital Energies: ', Energies['orbitalEnergies'])
#         TwoMeshStart=1

        if ( (len(X)!=len(Xf)) and (SCFcount>TwoMeshStart)  ):
            twoMesh=True
        else:
            twoMesh=False
#             
#         twoMesh=True
#         rprint(rank,"Setting twoMesh=True in SCF iteration so that the Hartree energy is computed more accurately.  Might not be necessary, but want to check results for C60 before convergence.")
            
            
        SCFindex = SCFcount
        if SCFcount>TwoMeshStart:
            SCFindex = SCFcount - TwoMeshStart
            
        if SCFcount==1:
            ## For the greedy approach, let the density start as the sum of wavefunctions.
            if GPUpresent: MOVEDATA.callRemoveVectorFromDevice(orbitals)
            orbitals, Energies['orbitalEnergies'] = sortByEigenvalue(orbitals,Energies['orbitalEnergies'])
            if GPUpresent: MOVEDATA.callCopyVectorToDevice(orbitals)
            fermiObjectiveFunction = fermiObjectiveFunctionClosure(Energies,nElectrons)  
            upperBound=1
#             lowerBoundIdx = int(np.floor(nElectrons/2))-1   
            lowerBound =  Energies['orbitalEnergies'][0]
            eF = brentq(fermiObjectiveFunction, lowerBound, upperBound, xtol=1e-14)
            if verbosity>0: rprint(rank,'Fermi energy: %f'%eF)
            
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
                
                eigenvalueResiduals=np.ones_like(residuals)  # reset eigenvalue residuals
                residuals = np.ones_like(Energies['orbitalEnergies'])
     
        
    
        ### Compute Veff
        """ 
        Compute new electron-electron potential and update pointwise potential values 
        """
        
        # interpolate density to fine mesh for computing hartree potential
#         rprint(rank, pointsPerCell_coarse)
#         rprint(rank, pointsPerCell_fine)
#         rprint(rank, len(X))
#         rprint(rank, len(Xf))
#         exit(-1)

#         if len(X) != len(Xf):
        if twoMesh==True:
            rprint(rank, "Interpolating density from %i to %i point mesh." %(len(X),len(Xf)))
#             start=time.time()
#             RHOf = interpolateBetweenTwoMeshes(X, Y, Z, RHO, pointsPerCell_coarse,
#                                                    Xf, Yf, Zf, pointsPerCell_fine)
#             end=time.time()
#             rprint(rank, "Original interpolation time: ", end-start)
#             
            numberOfCells=len(pointsPerCell_coarse)
#             start=time.time()
#             rprint(rank, "pointsPerCell_coarse = ", pointsPerCell_coarse[0:5])
#             rprint(rank, "pointsPerCell_fine = ", pointsPerCell_fine[0:5])
#             if numberOfCells
#             rprint(rank, "Type: ", pointsPerCell_fine.dtype)
            RHOf = interpolation_wrapper.callInterpolator(X,  Y,  Z,  RHO, pointsPerCell_coarse,
                                                           Xf, Yf, Zf, pointsPerCell_fine, 
                                                           numberOfCells, order, GPUpresent)
            
            
            
            
#             end=time.time()
#             comm.barrier() 
#             rprint(rank, "External interpolation time: ", end-start)
#             rprint(rank, "Difference: ", np.max( np.abs(RHOf-RHOf2)))
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
# #                     rprint(rank, i,RHOf[i], RHOf2[i])
#                     
#             rprint(rank, "Number of zeros: ", fCount, f2Count)
#             exit(-1)
#             rprint(rank, "RHOf = ", RHOf)
#             rprint(rank, "RHOf2 = ", RHOf2)
#             
#             
#             start=0
#             for i in range(numberOfCells):
#                 if pointsPerCell_fine[i]==512:
#                     for j in range(512):
#                         rprint(rank, RHOf[start+j], RHOf2[start+j])
#                     exit(-1)
#                 start+=int(pointsPerCell_fine[i])
#             exit(-1)
        else:
#             rprint(rank, "WHY IS LEN(X)=LEN(Xf)?")
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
            if verbosity>0: rprint(rank,'Convolution time: ', time.time()-start)
        else:    
            if singularityHandling=='skipping':
                if regularize==False:
                    if verbosity>0: rprint(rank,"Using singularity skipping in Hartree solve.")
                    kernelName = "coulomb"
                    numberOfKernelParameters=1
                    kernelParameters=np.array([0.0])
                elif regularize==True:
                    if verbosity>0: rprint(rank,"Using regularize coulomb kernel with epsilon = ", epsilon)
                    kernelName = "regularized-coulomb"
                    numberOfKernelParameters=1
                    kernelParameters=np.array([epsilon])
                else:
                    rprint(rank, "What should regularize be in SCF?")
                    exit(-1)
            elif singularityHandling=='subtraction':
                
                if regularize==False:                    
                    if verbosity>0: rprint(rank,"Using singularity subtraction in Hartree solve.")
                    kernelName = "coulomb"
                    numberOfKernelParameters=1
                    kernelParameters=np.array([gaussianAlpha])
                elif regularize==True:
                    if verbosity>0: rprint(rank,"Using SS and regularization for Hartree solve.")
                    kernelName="regularized-coulomb"
                    numberOfKernelParameters=2
                    kernelParameters=np.array([gaussianAlpha,epsilon])
                    
            else: 
                rprint(rank,"What should singularityHandling be?")
                return
            start = MPI.Wtime()
            
            
#             rprint(0, "Rank %i calling treecode through wrapper..." %(rank))
            
            treecode_verbosity=0
            
            if twoMesh:  # idea: only turn on the two mesh if beyond 4 SCF iterations
                numSources = len(Xf)
                sourceX=Xf
                sourceY=Yf
                sourceZ=Zf
                sourceRHO=RHOf
                sourceW=Wf
            else: 
                numSources = len(X)
                sourceX=X
                sourceY=Y
                sourceZ=Z
                sourceRHO=RHO
                sourceW=W
                
#             singularityHandling="skipping"
#             rprint(rank, "Forcing the Hartree solve to use singularity skipping.")

            if verbosity>0: rprint(rank,"Performing Hartree solve on %i mesh points" %numSources)
#             rprint(rank,"Coarse order ", order)
#             rprint(rank,"Fine order   ", fine_order)
#             approximation = BT.Approximation.LAGRANGE
#             singularity   = BT.Singularity.SUBTRACTION
#             computeType   = BT.ComputeType.PARTICLE_CLUSTER
#             
            kernel = BT.Kernel.COULOMB
            if singularityHandling=="subtraction":
                singularity=BT.Singularity.SUBTRACTION
            elif singularityHandling=="skipping":
                singularity=BT.Singularity.SKIPPING
            else:
                rprint(rank, "What should singularityHandling be?")
                exit(-1)
            
            if approximationName=="lagrange":
                approximation=BT.Approximation.LAGRANGE
            elif approximationName=="hermite":
                approximation=BT.Approximation.HERMITE
            else:
                rprint(rank, "What should approximationName be?")
                exit(-1)
            
            computeType=BT.ComputeType.PARTICLE_CLUSTER
                
    
            comm.barrier()
#             rprint(rank,"Using tighter treecode parameters for Hartree solve.")
            V_hartreeNew = BT.callTreedriver(  
                                            nPoints, numSources, 
                                            np.copy(X), np.copy(Y), np.copy(Z), np.copy(RHO), 
                                            np.copy(sourceX), np.copy(sourceY), np.copy(sourceZ), np.copy(sourceRHO), np.copy(sourceW),
                                            kernel, numberOfKernelParameters, kernelParameters, 
                                            singularity, approximation, computeType,
                                            treecodeOrder+0, theta-0.0, maxParNode, batchSize,
                                            GPUpresent, treecode_verbosity
                                            )
            

            if verbosity>0: rprint(rank,'Convolution time: ', MPI.Wtime()-start)
        
        
         
        """ 
        Compute the new orbital and total energies  
        """
        
        ## Energy update after computing Vhartree
        
        comm.barrier()  
        Energies["Repulsion"] = global_dot(RHO, Vext_local*W, comm)  
        Energies['Ehartree'] = 1/2*global_dot(W, RHO * V_hartreeNew, comm)
        
        rprint(rank,"Initial Hartree, nuclear, and repulsion energies: % .6f, % .6f, % .6f Ha" %(Energies["Ehartree"], Energies["Enuclear"], Energies["Repulsion"]))
        if abortAfterInitialHartree==True:
            Energies["Repulsion"] = global_dot(RHO, Vext_local*W, comm)
        
            Energies['totalElectrostatic'] = Energies["Ehartree"] + Energies["Enuclear"] + Energies["Repulsion"]
            rprint(rank,"Energies['Repulsion'] after initial convolution: ", Energies['Repulsion'])
            rprint(rank,"Energies['Ehartree'] after initial convolution: ", Energies['Ehartree'])
#             rprint(rank,"Electrostatics error after initial convolution: ", Energies['totalElectrostatic']-referenceEnergies["Eelectrostatic"])
            exit(-1)
            return np.zeros(nPoints)
        
        
        
        exchangeOutput = exchangeFunctional.compute(NLCC_RHO)
        correlationOutput = correlationFunctional.compute(NLCC_RHO) # For NLCC, evaluate the xc functionals on RHO+CORECHARGERHO.  For systems without NLCC, CORECHARGERHO==0 so it has no effect.
#         Energies['Ex'] = np.sum( W * RHO * np.reshape(exchangeOutput['zk'],np.shape(RHO)) )
#         Energies['Ec'] = np.sum( W * RHO * np.reshape(correlationOutput['zk'],np.shape(RHO)) )
        
        Energies['Ex'] = global_dot( W, NLCC_RHO * np.reshape(exchangeOutput['zk'],np.shape(RHO)), comm )
        Energies['Ec'] = global_dot( W, NLCC_RHO * np.reshape(correlationOutput['zk'],np.shape(RHO)), comm )
        
        Vx = np.reshape(exchangeOutput['vrho'],np.shape(RHO))
        Vc = np.reshape(correlationOutput['vrho'],np.shape(RHO))
        
#         Energies['Vx'] = np.sum(W * RHO * Vx)
#         Energies['Vc'] = np.sum(W * RHO * Vc)

#         Energies['Vx'] = global_dot(W, NLCC_RHO * Vx,comm)
#         Energies['Vc'] = global_dot(W, NLCC_RHO * Vc,comm)
        
        Energies['Vx'] = global_dot(W, RHO * Vx,comm)
        Energies['Vc'] = global_dot(W, RHO * Vc,comm)
        
        Veff_local = V_hartreeNew + Vx + Vc + Vext_local + gaugeShift
        
        
        if SCFcount==1: # generate initial guesses for eigenvalues
            Energies['Eold']=-10
            for m in range(nOrbitals):
                Energies['orbitalEnergies'][m]=-1.0
                
                

        ## Sort by eigenvalue
        
        if GPUpresent: 
            if verbosity>0: rprint(rank,"About to call MOVEDATA before sorting by eigenvalue.")
            MOVEDATA.callRemoveVectorFromDevice(orbitals)
        orbitals, Energies['orbitalEnergies'] = sortByEigenvalue(orbitals,Energies['orbitalEnergies'])
        if GPUpresent: 
            MOVEDATA.callCopyVectorToDevice(orbitals) 
            if verbosity>0: rprint(rank,"Completed calls to MOVEDATA after sorting by eigenvalue.")
        
        ## Solve the eigenvalue problem
        if SCFcount>1:
            fermiObjectiveFunction = fermiObjectiveFunctionClosure(Energies,nElectrons)        
            upperBound=1
#             lowerBoundIdx = int(np.floor(nElectrons/2))-1   
            lowerBound =  Energies['orbitalEnergies'][0]
            eF = brentq(fermiObjectiveFunction, lowerBound, upperBound, xtol=1e-14)
            if verbosity>0: rprint(rank,'Fermi energy: %f'%eF)
            exponentialArg = (Energies['orbitalEnergies']-eF)/Sigma
            previousOccupations = 2*1/(1+np.exp( exponentialArg ) )
        elif SCFcount==1: 
            previousOccupations = np.ones(nOrbitals)
            
        numPasses=1
        for passID in range(numPasses):
            rprint(rank,"GI pass %i" %(passID+1) )
            for m in range(nOrbitals): 
                if GPUpresent: MOVEDATA.callRemoveVectorFromDevice(orbitals)
                if GPUpresent: MOVEDATA.callCopyVectorToDevice(orbitals) 
    #             if previousOccupations[m] > 1e-20:
                if (  (previousOccupations[m] > 1e-20) or (SCFcount<8) ):  # 
                    if verbosity>0: rprint(rank,'Working on orbital %i' %m)
                    if verbosity>0: rprint(rank,'MEMORY USAGE: %i' %resource.getrusage(resource.RUSAGE_SELF).ru_maxrss )
                    
                    eigenvalueResiduals=np.ones_like(residuals)           
                    greenIterationsCount=1
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
                                   'singleWavefunctionOrthogonalization':True} 
                    
                    n,M = np.shape(orbitals)
                    resNorm=1.0 
                     
                    
    #                 ## Use previous eigenvalue to generate initial guess
    #                 if SCFcount==1:
    #                     gi_args['updateEigenvalue']=False
    #                     resNormWithoutEig=1 
    #                     orbitals[m,:] = np.random.rand(nPoints)
    #                     if m==0:
    #                         previousEigenvalue=-10
    #                     else:
    #                         previousEigenvalue=Energies['orbitalEnergies'][m-1]
    #                        
    #                     while resNormWithoutEig>1e-2:
    #                         Energies['orbitalEnergies'][m] = previousEigenvalue
    #                         psiIn = np.append( np.copy(orbitals[m,:]), Energies['orbitalEnergies'][m] )
    #                         greensIteration_FixedPoint, gi_args = greensIteration_FixedPoint_Closure(gi_args)
    #                         r = greensIteration_FixedPoint(psiIn, gi_args)
    #                         Energies['orbitalEnergies'][m] = previousEigenvalue
    #                         clenshawCurtisNorm = clenshawCurtisNormClosureWithoutEigenvalue(W)
    #                         resNormWithoutEig = clenshawCurtisNorm(r)
    #                         
    #                         rprint(rank, 'CC norm of residual vector: ', resNormWithoutEig)
    #                     rprint(rank, "Finished generating initial guess.\n\n")
    #                     gi_args['updateEigenvalue']=True
    
                    
                    comm.barrier()
                    if SCFcount==1:  
                        AndersonActivationTolerance=3e-2
                    elif SCFcount == TwoMeshStart:
                        AndersonActivationTolerance=3e-2
                    else:
                        AndersonActivationTolerance=3e-2
                    while ( (resNorm> max(AndersonActivationTolerance,scf_args['currentGItolerance'])) or (Energies['orbitalEnergies'][m]>0.0) ):
    #                 while resNorm>intraScfTolerance:
        #                 rprint(rank, 'MEMORY USAGE: ', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss )
        #                 GPUtil.showUtilization()
                        if verbosity>0: rprint(rank,'MEMORY USAGE: %i' %resource.getrusage(resource.RUSAGE_SELF).ru_maxrss )
                    
                        # Orthonormalize orbital m before beginning Green's iteration
        #                 n,M = np.shape(orbitals)
        #                 orthWavefunction = modifiedGramSchmidt_singleOrbital(orbitals,W,m, n, M)
        #                 orbitals[m,:] = np.copy(orthWavefunction)
        #                 rprint(rank, 'orbitals before: ',orbitals[1:5,m])
        #                 rprint(rank, 'greenIterationsCount before: ', greenIterationsCount)
                        psiIn = np.append( np.copy(orbitals[m,:]), Energies['orbitalEnergies'][m] )
             
                
                
                        oldEigenvalue = np.copy(Energies['orbitalEnergies'][m])
                        greensIteration_FixedPoint, gi_args = greensIteration_FixedPoint_Closure(gi_args)
                        r = greensIteration_FixedPoint(psiIn, gi_args)
                        newEigenvalue = np.copy(Energies['orbitalEnergies'][m])
                        eigenvalueDiff=np.abs(oldEigenvalue - newEigenvalue )
                        comm.barrier()
                        if verbosity>0: rprint(rank,'eigenvalueDiff = %f' %eigenvalueDiff)

                        
    
                        clenshawCurtisNorm = clenshawCurtisNormClosure(W)
                        resNorm = clenshawCurtisNorm(r)
                        
                        if verbosity>0: rprint(rank,'CC norm of residual vector: %f'%resNorm)
                        if eigenvalueDiff < resNorm/10:
                            resNorm = eigenvalueDiff
                            if verbosity>0: rprint(rank,'Using eigenvalueDiff: %f' %resNorm)
    
        
                    
                    psiOut = np.append(orbitals[m,:],Energies['orbitalEnergies'][m])
                    if verbosity>0: rprint(rank,'Power iteration tolerance met.  Beginning rootfinding now...') 
                    
                    
                    ## Begin Anderson Mixing on Wavefunction
                    Done = False
                    firstInputWavefunction=True
                    firstOutputWavefunction=True
                    inputWavefunctions = np.zeros((psiOut.size,1))
                    outputWavefunctions =  np.zeros((psiOut.size,1))
                    mixingStart = np.copy( gi_args['greenIterationsCount'] )
        
                    while Done==False:
                          
                        greenIterationsCount=gi_args["greenIterationsCount"]
                        if verbosity>0: rprint(rank,'MEMORY USAGE: %i'%resource.getrusage(resource.RUSAGE_SELF).ru_maxrss )
                          
                          
                        # Why is this mgs call happening?  Isn't  orbitals[m,:] already orthogonal after exiting the previous Green Iteration call? 
                        ### ACTUALLY THIS APPEARS TO BE IMPORTANT.... WHY? ###
                        ### OHHH, because after anderson mixing the resulting wavefunction won't still be orthogonal.  Apparently this *can* matter for convergence. 
                        ### Still though, how would this affect the first wavefunction?  It says the measured difference is 0.00e0.  And it only needs to be normalized, not orthogonalized. 
                        U=np.copy(orbitals[m])
                        if GPUpresent: MOVEDATA.callCopyVectorToDevice(U)
                        ORTH.callOrthogonalization(orbitals, U, W, m, GPUpresent)
                        if GPUpresent: MOVEDATA.callCopyVectorFromDevice(U)
                        orthWavefunction=np.copy(U)
                        
    #                     orthWavefunction = mgs(orbitals,W,m, comm)
    #                     diff=orthWavefunction-orbitals[m,:]
    #                     L2diff = np.sqrt(global_dot(W,diff**2,comm)) 
    #                     rprint(rank,"\n\n\n Just did the extra orthgonalization.  L2 difference between orbitals[m,:] and the newly orthogonalize vector: %1.2e" %L2diff)
                        orbitals[m,:] = np.copy(orthWavefunction)
                        psiIn = np.append( np.copy(orbitals[m,:]), Energies['orbitalEnergies'][m] )
                          
                          
                        ## Update input wavefunctions
                        if firstInputWavefunction==True:
                            inputWavefunctions[:,0] = np.copy(psiIn) # fill first column of inputWavefunctions
                            firstInputWavefunction=False
                        else:
                            if (greenIterationsCount-1-mixingStart)<GImixingHistoryCutoff:
                                inputWavefunctions = np.concatenate( ( inputWavefunctions, np.reshape(np.copy(psiIn), (psiIn.size,1)) ), axis=1)
    #                             rprint(rank, 'Concatenated inputWavefunction.  Now has shape: ', np.shape(inputWavefunctions))
          
                            else:
    #                             rprint(rank, 'Beyond GImixingHistoryCutoff.  Replacing column ', (greenIterationsCount-1-mixingStart)%GImixingHistoryCutoff)
                                inputWavefunctions[:,(greenIterationsCount-1-mixingStart)%GImixingHistoryCutoff] = np.copy(psiIn)
          
                        ## Perform one step of iterations
                        oldEigenvalue = np.copy(Energies['orbitalEnergies'][m])
                        greensIteration_FixedPoint, gi_args = greensIteration_FixedPoint_Closure(gi_args)
                        r = greensIteration_FixedPoint(psiIn,gi_args)
                        newEigenvalue = np.copy(Energies['orbitalEnergies'][m])
                        psiOut = np.append( gi_args["orbitals"][m,:], Energies['orbitalEnergies'][m])
                        clenshawCurtisNorm = clenshawCurtisNormClosure(W)
                        errorNorm = clenshawCurtisNorm(r)
                        if verbosity>0: rprint(rank,'Error Norm: %f' %errorNorm)
                        if errorNorm < scf_args['currentGItolerance']:
                            Done=True
                        eigenvalueDiff = np.abs(oldEigenvalue-newEigenvalue)
                        if verbosity>0: rprint(rank,'Eigenvalue Diff: %f' %eigenvalueDiff)
                        if ( (eigenvalueDiff<scf_args['currentGItolerance']/10) and (gi_args["greenIterationsCount"]>8) ): 
                            Done=True
                        if greenIterationsCount>50:
                            rprint(rank,"Terminating fixed point iteration for state %i at 50 iterations." %m) 
                            Done=True
                            
    #                     if ( (greenIterationsCount>20) and (Energies['orbitalEnergies'][m] > Energies['gaugeShift']) ):
    #                         rprint(rank,"Terminating fixed point iteration at 20 iterations because energy is still positive.")
    #                         Done=True
                            
                        
    #                     if ( (eigenvalueDiff < intraScfTolerance/10) and (gi_args['greenIterationsCount'] > 20) and ( ( SCFcount <2 ) or previousOccupations[m]<1e-4 ) ):  # must have tried to converge wavefunction. If after 20 iteration, allow eigenvalue tolerance to be enough. 
    #                         rprint(rank, 'Ending iteration because eigenvalue is converged.')
    #                         Done=True
                          
                          
                          
                        ## Update output wavefunctions
                          
                        if firstOutputWavefunction==True:
        #                     temp = np.append( orbitals[m,:], Energies['orbitalEnergies'][m])
                            outputWavefunctions[:,0] = np.copy(psiOut) # fill first column of outputWavefunctions
                            firstOutputWavefunction=False
                        else:
                            if (greenIterationsCount-1-mixingStart)<GImixingHistoryCutoff:
                                outputWavefunctions = np.concatenate( ( outputWavefunctions, np.reshape(np.copy(psiOut), (psiOut.size,1)) ), axis=1)
                            else:
                                outputWavefunctions[:,(greenIterationsCount-1-mixingStart)%GImixingHistoryCutoff] = np.copy(psiOut)
                          
                          
                          
                          
                        ## Compute next input wavefunctions
                        if verbosity>0: rprint(rank,'Anderson mixing on the orbital.')
                        GImixingParameter=0.5
                        andersonOrbital, andersonWeights = densityMixing.computeNewDensity(inputWavefunctions, outputWavefunctions, GImixingParameter,np.append(W,1.0), returnWeights=True)
                        Energies['orbitalEnergies'][m] = andersonOrbital[-1]
                        orbitals[m,:] = andersonOrbital[:-1] 
                        
                         
                        
                        if Energies['orbitalEnergies'][m]>0.0:
                            # Anderson mixing led to a positive eigenvalue.  This better be an unoccupied state.  Fixing it to the gauge shift value.
                            rprint(rank,"Anderson mixing led to a positive eigenvalue for %i state.  This better be an unoccupied state.  Fixing it to the gauge shift value." %m)
                            Energies['orbitalEnergies'][m] = Energies['gaugeShift']
                          
    #                     if ( (Done==False) and (greenIterationsCount%20==0) ):
    #                         rprint(rank,"Resetting history for state %i after iteration %i." %(m,greenIterationsCount))                         
    #                         firstInputWavefunction=True
    #                         firstOutputWavefunction=True
    #                         inputWavefunctions = np.zeros((psiOut.size,1))
    #                         outputWavefunctions =  np.zeros((psiOut.size,1))
    #                         mixingStart = np.copy( gi_args['greenIterationsCount'] ) 
    #       
          
                       
                    if verbosity>0: rprint(rank,'Used %i iterations for wavefunction %i' %(gi_args["greenIterationsCount"],m))
                else:
                    if verbosity>0: rprint(rank,"Not updating orbital %i because it is unoccupied." %m)
                
    
            
            ## Sort by eigenvalue
                    
            if GPUpresent: MOVEDATA.callRemoveVectorFromDevice(orbitals)
            orbitals, Energies['orbitalEnergies'] = sortByEigenvalue(orbitals,Energies['orbitalEnergies'])
            if GPUpresent: MOVEDATA.callCopyVectorToDevice(orbitals)
                
            
            fermiObjectiveFunction = fermiObjectiveFunctionClosure(Energies,nElectrons)        
            upperBound=1    
            lowerBoundIdx = int(np.floor(nElectrons/2))-1   
    #         lowerBound =  Energies['orbitalEnergies'][lowerBoundIdx]
            lowerBound =  Energies['orbitalEnergies'][0]
            eF = brentq(fermiObjectiveFunction, lowerBound, upperBound, xtol=1e-14)
            if verbosity>0: rprint(rank,'Fermi energy: ', eF)
            exponentialArg = (Energies['orbitalEnergies']-eF)/Sigma
            occupations = 2*1/(1+np.exp( exponentialArg ) )  # these are # of electrons, not fractional occupancy.  Hence the 2*
    
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
            

            if (SCFindex-1)<mixingHistoryCutoff:
                outputDensities = np.concatenate( (outputDensities, np.reshape(np.copy(newDensity), (nPoints,1))), axis=1)
#                 rprint(rank, 'Concatenated outputDensity.  Now has shape: ', np.shape(outputDensities))
            else:
#                 rprint(rank, 'Beyond mixingHistoryCutoff.  Replacing column ', (SCFcount-1)%mixingHistoryCutoff)
    #                                 rprint(rank, 'Shape of oldOrbitals[m,:]: ', np.shape(oldOrbitals[m,:]))
                outputDensities[:,(SCFindex-1)%mixingHistoryCutoff] = newDensity
            
            if SCFcount == TwoMeshStart:
                outputDensities = np.zeros((nPoints,1))         
                outputDensities[:,0] = np.copy(newDensity)
    
         
  
        if verbosity>0: 
            rprint(rank,'outputDensities[0,:] = ', outputDensities[0,:])
        
            rprint(rank,"Shape of input densities: ", np.shape(inputDensities))
            rprint(rank,"Shape of output densities: ", np.shape(outputDensities))
            rprint(rank,"\n\n\n\n\n")

        integratedDensity = global_dot( newDensity, W, comm )
        densityResidual = np.sqrt( global_dot( (newDensity-oldDensity)**2,W,comm ) ) / len(atoms)
        if verbosity>0: rprint(rank,'Integrated density: ', integratedDensity)
        if verbosity>0: rprint(rank,'Density Residual ', densityResidual)
        
        
#         ## Compute new Hartree potential and energy
# #         if twoMesh:  # idea: only turn on the two mesh if beyond 4 SCF iterations
# #             numSources = len(Xf)
# #             sourceX=Xf
# #             sourceY=Yf
# #             sourceZ=Zf
# #             sourceRHO=RHOf
# #             sourceW=Wf
# #         else: 
#         numSources = len(X)
#         sourceX=X
#         sourceY=Y
#         sourceZ=Z
#         sourceRHO=newDensity
#         sourceW=W
# 
#         if verbosity>0: rprint(rank,"Performing Hartree solve on %i mesh points" %numSources)
# #             rprint(rank,"Coarse order ", order)
# #             rprint(rank,"Fine order   ", fine_order)
# #             approximation = BT.Approximation.LAGRANGE
# #             singularity   = BT.Singularity.SUBTRACTION
# #             computeType   = BT.ComputeType.PARTICLE_CLUSTER
# #             
#         kernel = BT.Kernel.COULOMB
#         if singularityHandling=="subtraction":
#             singularity=BT.Singularity.SUBTRACTION
# #         elif singularityHandling=="skipping":
# #             singularity=BT.Singularity.SKIPPING
#         else:
#             rprint(rank, "What should singularityHandling be?")
#             exit(-1)
#         
#         if approximationName=="lagrange":
#             approximation=BT.Approximation.LAGRANGE
#         elif approximationName=="hermite":
#             approximation=BT.Approximation.HERMITE
#         else:
#             rprint(rank, "What should approximationName be?")
#             exit(-1)
#         
#         computeType=BT.ComputeType.PARTICLE_CLUSTER
#             
# 
#         comm.barrier()
#         V_hartreeNew = BT.callTreedriver(  
#                                         nPoints, numSources, 
#                                         np.copy(X), np.copy(Y), np.copy(Z), np.copy(newDensity), 
#                                         np.copy(sourceX), np.copy(sourceY), np.copy(sourceZ), np.copy(newDensity), np.copy(sourceW),
#                                         kernel, numberOfKernelParameters, kernelParameters, 
#                                         singularity, approximation, computeType,
#                                         treecodeOrder, theta, maxParNode, batchSize,
#                                         GPUpresent, treecode_verbosity
#                                         )
#         
# 
#         if verbosity>0: rprint(rank,'Hartree Convolution time: ', MPI.Wtime()-start)
#         
#         
#         
#         """ 
#         Compute the new orbital and total energies 
#         """
#         
#         ## Energy update after computing Vhartree
#         
#         comm.barrier()    
#         Energies['Ehartree'] = 1/2*global_dot(W, RHO * V_hartreeNew, comm)
        
        
        Energies["Repulsion"] = global_dot(newDensity, Vext_local*W, comm)
        
        Energies['Eband'] = np.sum( (Energies['orbitalEnergies']-Energies['gaugeShift']) * occupations)
        Energies['Etotal'] = Energies['Eband'] - Energies['Ehartree'] + Energies['Ex'] + Energies['Ec'] - Energies['Vx'] - Energies['Vc'] + Energies['Enuclear']
        Energies['totalElectrostatic'] = Energies["Ehartree"] + Energies["Enuclear"] + Energies["Repulsion"]
        
    
        for m in range(nOrbitals):
            if verbosity>0: rprint(rank,'Orbital %i error: %1.3e' %(m, Energies['orbitalEnergies'][m]-referenceEigenvalues[m]-Energies['gaugeShift']))
        
        
        energyResidual = abs( Energies['Etotal'] - Energies['Eold'] )/len(atoms)  # Compute the energyResidual for determining convergence
        Energies['Eold'] = np.copy(Energies['Etotal'])
        
        
        
        """
        Print results from current iteration
        """
    
        print_eigs_and_occupations(Energies['orbitalEnergies']-Energies['gaugeShift'], occupations, Energies['orbitalEnergies']-referenceEigenvalues[:nOrbitals]-Energies['gaugeShift'])
        rprint(rank,'Updated V_x:                               % .10f Ha' %Energies['Vx'])
        rprint(rank,'Updated V_c:                               % .10f Ha' %Energies['Vc'])
        rprint(rank,'Updated Band Energy:                       % .10f Ha, %.10e Ha' %(Energies['Eband'], Energies['Eband']-referenceEnergies['Eband']) )
    #         rprint(rank, 'Updated Kinetic Energy:                 %.10f H, %.10e Ha' %(Energies['kinetic'], Energies['kinetic']-Ekinetic) )
        rprint(rank,'Updated E_Hartree:                         % .10f H, %.10e Ha' %(Energies['Ehartree'], Energies['Ehartree']-referenceEnergies['Ehartree']) )
        rprint(rank,'Updated E_x:                               % .10f H, %.10e Ha' %(Energies['Ex'], Energies['Ex']-referenceEnergies['Eexchange']) )
        rprint(rank,'Updated E_c:                               % .10f H, %.10e Ha' %(Energies['Ec'], Energies['Ec']-referenceEnergies['Ecorrelation']) )
        rprint(rank,'Updated totalElectrostatic:                % .10f H, %.10e Ha' %(Energies['totalElectrostatic'], Energies['totalElectrostatic']-referenceEnergies["Eelectrostatic"]))
        rprint(rank,"Hartree, Nuclear, Repulsion:               % .6f, % .6f, % .6f Ha" %(Energies["Ehartree"], Energies["Enuclear"], Energies["Repulsion"]))
        rprint(rank,'Total Energy:                              % .10f H, %.10e Ha' %(Energies['Etotal'], Energies['Etotal']-referenceEnergies['Etotal']))
        rprint(rank,'Total Energy Per Atom:                     % .10f H, %.10e Ha' %(Energies['Etotal']/len(atoms), (Energies['Etotal']-referenceEnergies['Etotal'])/len(atoms) ))
        rprint(rank,'Energy Residual per atom:                  % .3e' %energyResidual)
        rprint(rank,'Density Residual per atom:                 % .3e\n\n'%densityResidual)
    
        scf_args['energyResidual']=energyResidual
        scf_args['densityResidual']=densityResidual
        
        
            
    #         if vtkExport != False:
    #             filename = vtkExport + '/mesh%03d'%(SCFcount-1) + '.vtk'
    #             Energies['Etotal']xportGridpoints(filename)
    
        printEachIteration=True
    
        if printEachIteration==True:
            header = ['Iteration', 'densityResidual', 'orbitalEnergies','bandEnergy', 'kineticEnergy', 
                      'exchangeEnergy', 'correlationEnergy', 'totalElectrostatic', 'totalEnergy', 'GItolerance']
         
            myData = [SCFcount, densityResidual, Energies['orbitalEnergies']-Energies['gaugeShift'], Energies['Eband'], Energies['kinetic'], 
                      Energies['Ex'], Energies['Ec'], Energies['totalElectrostatic'], Energies['Etotal'], scf_args['currentGItolerance']]

            
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
                
        if scf_args['GItolerancesIdx'] < scf_args['gradualSteps']-1: # GItolerancesIdx goes from 0 to gradualSteps-1
            scf_args['GItolerancesIdx']+=1
            scf_args['currentGItolerance'] = GItolerances[scf_args['GItolerancesIdx']]
            rprint(rank,'Reducing GI tolerance to ', scf_args['currentGItolerance'])
        
        
        ## Write the restart files
        ## COMMENTED OUT FOR NOW UNTIL RESTART CAPABILITY IS SUPPORTED FOR DOMAIN DECOMPOSITION APPROACH
         
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
            auxiliaryRestartData['GItolerancesIdx'] = scf_args['GItolerancesIdx']
            auxiliaryRestartData['eigenvalues'] = Energies['orbitalEnergies']
            auxiliaryRestartData['Eold'] = Energies['Eold']
     
            np.save(auxiliaryFile, auxiliaryRestartData)
        except FileNotFoundError as e:
            rprint(rank,"FileNotFoundError: ", e)
            exit(-1)
#             pass
                
        
        
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
    return scfFixedPoint, scf_args


if __name__=="__main__":
    
    


    eigs = -1*np.random.rand(10)
    occs = np.random.rand(10)
    print_eigs_and_occupations(eigs,occs)


