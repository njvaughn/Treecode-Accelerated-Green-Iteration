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


from fermiDiracDistribution import computeOccupations
import densityMixingSchemes as densityMixing
import treecodeWrappers_distributed as treecodeWrappers
from orthogonalizationRoutines import *
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
    
def sortByEigenvalue(orbitals,orbitalEnergies):
    newOrder = np.argsort(orbitalEnergies)
    oldEnergies = np.copy(orbitalEnergies)
    for m in range(len(orbitalEnergies)):
        orbitalEnergies[m] = oldEnergies[newOrder[m]]
    rprint(rank,'Sorted eigenvalues: ', orbitalEnergies)
    rprint(rank,'New order: ', newOrder)
    
    newOrbitals = np.zeros_like(orbitals)
    for m in range(len(orbitalEnergies)):
        newOrbitals[:,m] = orbitals[:,newOrder[m]]            
   
    return newOrbitals, orbitalEnergies
      
def scfFixedPointClosure(scf_args): 
    
    def scfFixedPoint(RHO,scf_args, abortAfterInitialHartree=False):
        
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
        order=scf_args['order']
        fine_order=scf_args['fine_order']
        regularize=scf_args['regularize']
        epsilon=scf_args['epsilon']
        
        GItolerances = np.logspace(np.log10(initialGItolerance),np.log10(finalGItolerance),gradualSteps)
#         scf_args['GItolerancesIdx']=0
        
        scf_args['currentGItolerance']=GItolerances[scf_args['GItolerancesIdx']]
        rprint(rank,"Current GI toelrance: ", scf_args['currentGItolerance'])
        
        GImixingHistoryCutoff = 10
         
        SCFcount += 1
        rprint(rank,'\nSCF Count ', SCFcount)
        rprint(rank,'Orbital Energies: ', Energies['orbitalEnergies'])
        
        
        if SCFcount>1:
            
    
            if (SCFcount-1)<mixingHistoryCutoff:
                inputDensities = np.concatenate( (inputDensities, np.reshape(RHO, (nPoints,1))), axis=1)
            else:
                inputDensities[:,(SCFcount-1)%mixingHistoryCutoff] = np.copy(RHO)
        
     
        
    
        ### Compute Veff
        """ 
        Compute new electron-electron potential and update pointwise potential values 
        """
        
        
        
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
            
            verbosity=0
#             singularityHandling="skipping"
#             print("Forcing the Hartree solve to use singularity skipping.")
            V_hartreeNew = treecodeWrappers.callTreedriver(nPoints, nPoints, 
                                                           np.copy(X), np.copy(Y), np.copy(Z), np.copy(RHO), 
                                                           np.copy(X), np.copy(Y), np.copy(Z), np.copy(RHO), np.copy(W),
                                                           kernelName, numberOfKernelParameters, kernelParameters, singularityHandling, approximationName,
                                                           treecodeOrder, theta, maxParNode, batchSize, GPUpresent, verbosity)
            
#             V_hartreeNew += 2.0*np.pi*gaussianAlpha*gaussianAlpha*RHO
            
            
            Times['timePerConvolution'] = MPI.Wtime()-start
            rprint(rank,'Convolution time: ', MPI.Wtime()-start)
        
      
        
        
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
            for m in range(nOrbitals):
                Energies['orbitalEnergies'][m]=-1.0
# #             orbitals, Energies['orbitalEnergies'] = sortByEigenvalue(orbitals, Energies['orbitalEnergies'])
# #             for m in range(nOrbitals):
# #                 if Energies['orbitalEnergies'][m] > 0:
# #                     Energies['orbitalEnergies'][m] = -0.5
        
        
        ## Sort by eigenvalue
        orbitals, Energies['orbitalEnergies'] = sortByEigenvalue(orbitals,Energies['orbitalEnergies'])
           
        
        ## Solve the eigenvalue problem
        if SCFcount>1:
            fermiObjectiveFunction = fermiObjectiveFunctionClosure(Energies,nElectrons)        
            eF = brentq(fermiObjectiveFunction, Energies['orbitalEnergies'][0], 1, xtol=1e-14)
            rprint(rank,'Fermi energy: %f'%eF)
            exponentialArg = (Energies['orbitalEnergies']-eF)/Sigma
            previousOccupations = 2*1/(1+np.exp( exponentialArg ) )
        elif SCFcount==1: 
            previousOccupations = np.ones(nOrbitals)
        for m in range(nOrbitals): 
            if previousOccupations[m] > 1e-20:
                rprint(rank,'Working on orbital %i' %m)
                rprint(rank,'MEMORY USAGE: %i' %resource.getrusage(resource.RUSAGE_SELF).ru_maxrss )
                
        #                         
                            
                greenIterationsCount=1
#                 print("Forcing singularityHandling to be skipping for Green's iteration.")
                gi_args = {'orbitals':orbitals,'oldOrbitals':oldOrbitals, 'Energies':Energies, 'Times':Times, 'Veff_local':Veff_local, 
                               'symmetricIteration':symmetricIteration,'GPUpresent':GPUpresent,
                               'singularityHandling':singularityHandling, 'approximationName':approximationName,
                               'treecode':treecode,'treecodeOrder':treecodeOrder,'theta':theta, 'maxParNode':maxParNode,'batchSize':batchSize,
                               'nPoints':nPoints, 'm':m, 'X':X,'Y':Y,'Z':Z,'W':W,'Xf':Xf,'Yf':Yf,'Zf':Zf,'Wf':Wf,'gradientFree':gradientFree,
                               'SCFcount':SCFcount,'greenIterationsCount':greenIterationsCount,'residuals':residuals,
                               'greenIterationOutFile':greenIterationOutFile,
                               'referenceEigenvalues':referenceEigenvalues,
                               'updateEigenvalue':True,
                               'coreRepresentation':coreRepresentation,
                               'atoms':atoms,
                               'order':order,
                               'fine_order':fine_order,
                               'regularize':regularize, 'epsilon':epsilon } 
                
                n,M = np.shape(orbitals)
                resNorm=1.0 
                
#                 orthWavefunction = modifiedGramSchmidt_singleOrbital(orbitals,W,m, n, M)
#                 orbitals[:,m] = np.copy(orthWavefunction)
                
                
                
#                 ## Use previous eigenvalue to generate initial guess
#                 if SCFcount==1:
#                     gi_args['updateEigenvalue']=False
#                     resNormWithoutEig=1 
#                     orbitals[:,m] = np.random.rand(nPoints)
#                     if m==0:
#                         previousEigenvalue=-10
#                     else:
#                         previousEigenvalue=Energies['orbitalEnergies'][m-1]
#                        
#                     while resNormWithoutEig>1e-2:
#                         Energies['orbitalEnergies'][m] = previousEigenvalue
#                         psiIn = np.append( np.copy(orbitals[:,m]), Energies['orbitalEnergies'][m] )
#                         greensIteration_FixedPoint, gi_args = greensIteration_FixedPoint_Closure(gi_args)
#                         r = greensIteration_FixedPoint(psiIn, gi_args)
#                         Energies['orbitalEnergies'][m] = previousEigenvalue
#                         clenshawCurtisNorm = clenshawCurtisNormClosureWithoutEigenvalue(W)
#                         resNormWithoutEig = clenshawCurtisNorm(r)
#                         
#                         print('CC norm of residual vector: ', resNormWithoutEig)
#                     print("Finished generating initial guess.\n\n")
#                     gi_args['updateEigenvalue']=True

                
                comm.barrier()
                while ( (resNorm> max(1e-4,scf_args['currentGItolerance'])) or (Energies['orbitalEnergies'][m]>0.0) ):
#                 while resNorm>intraScfTolerance:
    #                 print('MEMORY USAGE: ', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss )
    #                 GPUtil.showUtilization()
                    rprint(rank,'MEMORY USAGE: %i' %resource.getrusage(resource.RUSAGE_SELF).ru_maxrss )
                
                    # Orthonormalize orbital m before beginning Green's iteration
    #                 n,M = np.shape(orbitals)
    #                 orthWavefunction = modifiedGramSchmidt_singleOrbital(orbitals,W,m, n, M)
    #                 orbitals[:,m] = np.copy(orthWavefunction)
    #                 print('orbitals before: ',orbitals[1:5,m])
    #                 print('greenIterationsCount before: ', greenIterationsCount)
                    psiIn = np.append( np.copy(orbitals[:,m]), Energies['orbitalEnergies'][m] )
         
            
            
                    oldEigenvalue = np.copy(Energies['orbitalEnergies'][m])
                    greensIteration_FixedPoint, gi_args = greensIteration_FixedPoint_Closure(gi_args)
                    r = greensIteration_FixedPoint(psiIn, gi_args)
                    newEigenvalue = np.copy(Energies['orbitalEnergies'][m])
                    eigenvalueDiff=np.abs(oldEigenvalue - newEigenvalue )
                    comm.barrier()
                    rprint(rank,'eigenvalueDiff = %f' %eigenvalueDiff)
                    
    #                 print('greenIterationsCount after: ', greenIterationsCount)
    #                 print('gi_args greenIterationsCount after: ', gi_args["greenIterationsCount"])
    
    #                 print('orbitals after: ',orbitals[1:5,m])
    #                 print('gi_args orbitals after: ',gi_args["orbitals"][1:5,m])
                    clenshawCurtisNorm = clenshawCurtisNormClosure(W)
                    resNorm = clenshawCurtisNorm(r)
                    
                    rprint(rank,'CC norm of residual vector: %f'%resNorm)
                    if eigenvalueDiff < resNorm/10:
                        resNorm = eigenvalueDiff
                        rprint(rank,'Using eigenvalueDiff: %f' %resNorm)

    
                
                psiOut = np.append(orbitals[:,m],Energies['orbitalEnergies'][m])
                rprint(rank,'Power iteration tolerance met.  Beginning rootfinding now...') 
    #             psiIn = np.copy(psiOut)
#                 tol=intraScfTolerance
                
                
                ## Begin Anderson Mixing on Wavefunction
                Done = False
#                 Done = True
                firstInputWavefunction=True
                firstOutputWavefunction=True
                inputWavefunctions = np.zeros((psiOut.size,1))
                outputWavefunctions =  np.zeros((psiOut.size,1))
                mixingStart = np.copy( gi_args['greenIterationsCount'] )
                
                
     
    
                while Done==False:
                      
                      
                    greenIterationsCount=gi_args["greenIterationsCount"]
                    rprint(rank,'MEMORY USAGE: %i'%resource.getrusage(resource.RUSAGE_SELF).ru_maxrss )
                      
                    orthWavefunction = modifiedGramSchmidt_singleOrbital(orbitals,W,m, n, M, comm)
                    orbitals[:,m] = np.copy(orthWavefunction)
                    psiIn = np.append( np.copy(orbitals[:,m]), Energies['orbitalEnergies'][m] )
                      
                      
                    ## Update input wavefunctions
                    if firstInputWavefunction==True:
                        inputWavefunctions[:,0] = np.copy(psiIn) # fill first column of inputWavefunctions
                        firstInputWavefunction=False
                    else:
                        if (greenIterationsCount-1-mixingStart)<GImixingHistoryCutoff:
                            inputWavefunctions = np.concatenate( ( inputWavefunctions, np.reshape(np.copy(psiIn), (psiIn.size,1)) ), axis=1)
#                             print('Concatenated inputWavefunction.  Now has shape: ', np.shape(inputWavefunctions))
      
                        else:
#                             print('Beyond GImixingHistoryCutoff.  Replacing column ', (greenIterationsCount-1-mixingStart)%GImixingHistoryCutoff)
                            inputWavefunctions[:,(greenIterationsCount-1-mixingStart)%GImixingHistoryCutoff] = np.copy(psiIn)
      
                    ## Perform one step of iterations
                    oldEigenvalue = np.copy(Energies['orbitalEnergies'][m])
    #                 print('Before GI Energies:[orbitalEnergies] ', Energies['orbitalEnergies'])
                    greensIteration_FixedPoint, gi_args = greensIteration_FixedPoint_Closure(gi_args)
                    r = greensIteration_FixedPoint(psiIn,gi_args)
    #                 print('After GI Energies:[orbitalEnergies] ', Energies['orbitalEnergies'])
                    newEigenvalue = np.copy(Energies['orbitalEnergies'][m])
                    psiOut = np.append( gi_args["orbitals"][:,m], Energies['orbitalEnergies'][m])
                    clenshawCurtisNorm = clenshawCurtisNormClosure(W)
                    errorNorm = clenshawCurtisNorm(r)
                    rprint(rank,'Error Norm: %f' %errorNorm)
                    if errorNorm < scf_args['currentGItolerance']:
                        Done=True
                    eigenvalueDiff = np.abs(oldEigenvalue-newEigenvalue)
                    rprint(rank,'Eigenvalue Diff: %f' %eigenvalueDiff)
                    if ( (eigenvalueDiff<scf_args['currentGItolerance']/20) and (gi_args["greenIterationsCount"]>8) ): 
                        Done=True
#                     if ( (eigenvalueDiff < intraScfTolerance/10) and (gi_args['greenIterationsCount'] > 20) and ( ( SCFcount <2 ) or previousOccupations[m]<1e-4 ) ):  # must have tried to converge wavefunction. If after 20 iteration, allow eigenvalue tolerance to be enough. 
#                         print('Ending iteration because eigenvalue is converged.')
#                         Done=True
                      
                      
                      
                    ## Update output wavefunctions
                      
                    if firstOutputWavefunction==True:
    #                     temp = np.append( orbitals[:,m], Energies['orbitalEnergies'][m])
                        outputWavefunctions[:,0] = np.copy(psiOut) # fill first column of outputWavefunctions
                        firstOutputWavefunction=False
                    else:
                        if (greenIterationsCount-1-mixingStart)<GImixingHistoryCutoff:
    #                         temp = np.append( orbitals[:,m], Energies['orbitalEnergies'][m])
                            outputWavefunctions = np.concatenate( ( outputWavefunctions, np.reshape(np.copy(psiOut), (psiOut.size,1)) ), axis=1)
#                             print('Concatenated outputWavefunction.  Now has shape: ', np.shape(outputWavefunctions))
                        else:
#                             print('Beyond GImixingHistoryCutoff.  Replacing column ', (greenIterationsCount-1-mixingStart)%GImixingHistoryCutoff)
    #                         temp = np.append( orbitals[:,m], Energies['orbitalEnergies'][m])
                            outputWavefunctions[:,(greenIterationsCount-1-mixingStart)%GImixingHistoryCutoff] = np.copy(psiOut)
                      
                      
                      
                      
                    ## Compute next input wavefunctions
                    rprint(rank,'Anderson mixing on the orbital.')
                    GImixingParameter=0.5
                    andersonOrbital, andersonWeights = densityMixing.computeNewDensity(inputWavefunctions, outputWavefunctions, GImixingParameter,np.append(W,1.0), returnWeights=True)
                    Energies['orbitalEnergies'][m] = andersonOrbital[-1]
                    orbitals[:,m] = andersonOrbital[:-1]
                      
                      
      
      
                   
    #             print('Used %i iterations for wavefunction %i' %(greenIterationsCount,m))
                rprint(rank,'Used %i iterations for wavefunction %i' %(gi_args["greenIterationsCount"],m))
            else:
                rprint(rank,"Not updating orbital %i because it is unoccupied." %m)
            
            
# #                # Method that uses Scipy Anderson
#                 Done = False
#                 while Done==False:
#                     try:
#                         # Call anderson mixing on the Green's iteration fixed point function
#           
#                         # Orthonormalize orbital m before beginning Green's iteration
#                         n,M = np.shape(orbitals)
#                         orthWavefunction = modifiedGramSchmidt_singleOrbital(orbitals,W,m, n, M)
#                         orbitals[:,m] = np.copy(orthWavefunction)
#                            
#                         psiIn = np.append( np.copy(orbitals[:,m]), Energies['orbitalEnergies'][m] )
#           
#                              
#                         ### Anderson Options
#                         clenshawCurtisNorm = clenshawCurtisNormClosure(W)
#                         method='anderson'
# #                         jacobianOptions={'alpha':1.0, 'M':10, 'w0':0.01} 
#                         jacobianOptions={'alpha':1.0, 'M':5, 'w0':0.01} 
#                         solverOptions={'fatol':intraScfTolerance, 'tol_norm':clenshawCurtisNorm, 'jac_options':jacobianOptions,'maxiter':1000, 'line_search':None, 'disp':True}
# #                         solverOptions={'fatol':intraScfTolerance, 'tol_norm':clenshawCurtisNorm, 'jac_options':jacobianOptions,'maxiter':1000, 'disp':True}
#           
#                           
#                         print('Calling scipyRoot with %s method' %method)
#                         sol = scipyRoot(greensIteration_FixedPoint,psiIn, args=gi_args, method=method, options=solverOptions)
#                         print(sol.success)
#                         print(sol.message)
#                         psiOut = sol.x
#                         Done = True
#                     except Exception:
# #                         print('Not converged.  What to do?')
# #                         return
#                         if np.abs(gi_args['eigenvalueDiff']) < tol/10:
#                             print("Rootfinding didn't converge but eigenvalue is converged.  Exiting because this is probably due to degeneracy in the space.")
#         #                         targets = tree.extractPhi(m)
#                             psiOut = np.append(orbitals[:,m], Energies['orbitalEnergies'][m])
#                             Done=True
#                         else:
#                             print('Not converged.  What to do?')
#                             return
#                 orbitals[:,m] = np.copy(psiOut[:-1])
#                 Energies['orbitalEnergies'][m] = np.copy(psiOut[-1])
#                    
#                 print('Used %i iterations for wavefunction %i' %(greenIterationsCount,m))
#             
    
        
        ## Sort by eigenvalue
        
        orbitals, Energies['orbitalEnergies'] = sortByEigenvalue(orbitals,Energies['orbitalEnergies'])
        
        fermiObjectiveFunction = fermiObjectiveFunctionClosure(Energies,nElectrons)        
        eF = brentq(fermiObjectiveFunction, Energies['orbitalEnergies'][0], 1, xtol=1e-14)
        rprint(rank,'Fermi energy: ', eF)
        exponentialArg = (Energies['orbitalEnergies']-eF)/Sigma
        occupations = 2*1/(1+np.exp( exponentialArg ) )  # these are # of electrons, not fractional occupancy.  Hence the 2*
    
    #         occupations = computeOccupations(Energies['orbitalEnergies'], nElectrons, Temperature)
        rprint(rank,'Occupations: ', occupations)
        Energies['Eband'] = np.sum( (Energies['orbitalEnergies']-Energies['gaugeShift']) * occupations)
    
     
    
        
    
        
    
        oldDensity = np.copy(RHO)
        
        RHO = np.zeros(nPoints)
        for m in range(nOrbitals):
            RHO += orbitals[:,m]**2 * occupations[m]
        newDensity = np.copy(RHO)
        
        print("Integral of old RHO ", global_dot( oldDensity,W,comm ) )
        print("Integral of new RHO ", global_dot( newDensity,W,comm ) )
        
        print("NORMALIZING NEW RHO")
        densityIntegral=global_dot( newDensity,W,comm )
        newDensity *= nElectrons/densityIntegral
        
    
        if SCFcount==1: # not okay anymore because output density gets reset when tolerances get reset.
            outputDensities[:,0] = np.copy(newDensity)
#             scf_args['outputDensities']=outputDensities
        else:
            
    #             outputDensities = np.concatenate( ( outputDensities, np.reshape(np.copy(newDensity), (nPoints,1)) ), axis=1)
            
            if (SCFcount-1)<mixingHistoryCutoff:
                outputDensities = np.concatenate( (outputDensities, np.reshape(np.copy(newDensity), (nPoints,1))), axis=1)
#                 print('Concatenated outputDensity.  Now has shape: ', np.shape(outputDensities))
            else:
#                 print('Beyond mixingHistoryCutoff.  Replacing column ', (SCFcount-1)%mixingHistoryCutoff)
    #                                 print('Shape of oldOrbitals[:,m]: ', np.shape(oldOrbitals[:,m]))
                outputDensities[:,(SCFcount-1)%mixingHistoryCutoff] = newDensity
            
        rprint(rank,'outputDensities[0,:] = ', outputDensities[0,:])
#         print('outputDensities[:,0:3] = ', outputDensities[:,0:3])
        
    #         print('Sample of output densities:')
    #         print(outputDensities[0,:])    
#         integratedDensity = np.sum( newDensity*W )
        integratedDensity = global_dot( newDensity, W, comm )
#         densityResidual = np.sqrt( np.sum( (newDensity-oldDensity)**2*W ) )
        densityResidual = np.sqrt( global_dot( (newDensity-oldDensity)**2,W,comm ) )
        rprint(rank,'Integrated density: ', integratedDensity)
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
#                     Vext_nl += atom.V_nonlocal_pseudopotential_times_psi(X,Y,Z,orbitals[:,m],W,comm)
#                 Eext_nl += global_dot(orbitals[:,m], Vext_nl,comm)
#             Energies['Etotal'] += Eext_nl
    
        for m in range(nOrbitals):
            rprint(rank,'Orbital %i error: %1.3e' %(m, Energies['orbitalEnergies'][m]-referenceEigenvalues[m]-Energies['gaugeShift']))
        
        
        energyResidual = abs( Energies['Etotal'] - Energies['Eold'] )  # Compute the energyResidual for determining convergence
#         energyError = abs( Energies['Etotal'] - Energies['Eold'] )  # Compute the energyResidual for determining convergence
        Energies['Eold'] = np.copy(Energies['Etotal'])
        
        
        
        """
        Print results from current iteration
        """
    
        rprint(rank,'Orbital Energies: ', Energies['orbitalEnergies']) 
    
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
            print('Reducing GI tolerance to ', scf_args['currentGItolerance'])
        
        
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
    return scfFixedPoint, scf_args



