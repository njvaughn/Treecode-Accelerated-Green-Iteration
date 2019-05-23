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


from fermiDiracDistribution import computeOccupations
import sys
import resource


from greenIterationFixedPoint import greensIteration_FixedPoint_Closure
from orthogonalizationRoutines import *
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
    
    
def scfFixedPointClosure(scf_args): 
    
    def scfFixedPoint(RHO,scf_args):
        
        ## Unpack scf_args
        inputDensities = scf_args['inputDensities']
        outputDensities=scf_args['outputDensities']
        SCFcount = scf_args['SCFcount']
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
        
        
        if SCFcount>1:
            
    
            if (SCFcount-1)<mixingHistoryCutoff:
                inputDensities = np.concatenate( (inputDensities, np.reshape(RHO, (nPoints,1))), axis=1)
                print('Concatenated inputDensity.  Now has shape: ', np.shape(inputDensities))
            else:
                print('Beyond mixingHistoryCutoff.  Replacing column ', (SCFcount-1)%mixingHistoryCutoff)
    #                                 print('Shape of oldOrbitals[:,m]: ', np.shape(oldOrbitals[:,m]))
                inputDensities[:,(SCFcount-1)%mixingHistoryCutoff] = np.copy(RHO)
        
     
        
    
        ### Compute Veff
        """ 
        Compute new electron-electron potential and update pointwise potential values 
        """
    #         starthartreeConvolutionTime = timer()
        
        
        if GPUpresent==True:
            if treecode==False:
                V_hartreeNew = np.zeros(nPoints)
                densityInput = np.transpose( np.array([X,Y,Z,RHO,W]) )
                gpuHartreeGaussianSingularitySubract[blocksPerGrid, threadsPerBlock](densityInput,densityInput,V_hartreeNew,gaussianAlpha*gaussianAlpha)
            elif treecode==True:
                start = time.time()
                potentialType=2 
                V_hartreeNew = treecodeWrappers.callTreedriver(nPoints, nPoints, 
                                                               np.copy(X), np.copy(Y), np.copy(Z), np.copy(RHO), 
                                                               np.copy(X), np.copy(Y), np.copy(Z), np.copy(RHO), np.copy(W),
                                                               potentialType, gaussianAlpha, treecodeOrder, theta, maxParNode, batchSize)
                print('Convolution time: ', time.time()-start)
                
        elif GPUpresent==False:
            print('Error: not prepared for Hartree solve without GPU')
            return
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
            for m in range(nOrbitals):
#                 Energies['orbitalEnergies'][m] = -10
                Energies['orbitalEnergies'][m] = np.sum( W* orbitals[:,m]**2 * Veff) * (2/3) # Attempt to guess initial orbital energy without computing kinetic
            orbitals, Energies['orbitalEnergies'] = sortByEigenvalue(orbitals, Energies['orbitalEnergies'])
        
        
           
        
        ## Solve the eigenvalue problem
        
        for m in range(nOrbitals): 
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
                           'referenceEigenvalues':referenceEigenvalues   } 
            
            
            resNorm=1 
            while resNorm>1e-2:

                
            
                # Orthonormalize orbital m before beginning Green's iteration
                n,M = np.shape(orbitals)
                orthWavefunction = modifiedGramSchmidt_singleOrbital(orbitals,W,m, n, M)
                orbitals[:,m] = np.copy(orthWavefunction)
                psiIn = np.append( np.copy(orbitals[:,m]), Energies['orbitalEnergies'][m] )
    
        
        
                greensIteration_FixedPoint, gi_args = greensIteration_FixedPoint_Closure(gi_args)
                r = greensIteration_FixedPoint(psiIn, gi_args)
                clenshawCurtisNorm = clenshawCurtisNormClosure(W)
                resNorm = clenshawCurtisNorm(r)
                print('CC norm of residual vector: ', resNorm)

             
             
            print('Power iteration tolerance met.  Beginning rootfinding now...') 
            tol=intraScfTolerance
    
            Done = False
            while Done==False:
                try:
                    # Call anderson mixing on the Green's iteration fixed point function
    
                    # Orthonormalize orbital m before beginning Green's iteration
                    n,M = np.shape(orbitals)
                    orthWavefunction = modifiedGramSchmidt_singleOrbital(orbitals,W,m, n, M)
                    orbitals[:,m] = np.copy(orthWavefunction)
                     
                    psiIn = np.append( np.copy(orbitals[:,m]), Energies['orbitalEnergies'][m] )
    
                       
                    ### Anderson Options
                    clenshawCurtisNorm = clenshawCurtisNormClosure(W)
                    method='anderson'
                    jacobianOptions={'alpha':1.0, 'M':20, 'w0':0.01} 
                    solverOptions={'fatol':tol, 'tol_norm':clenshawCurtisNorm, 'jac_options':jacobianOptions,'maxiter':1000, 'line_search':None, 'disp':True}
    
                    
                    print('Calling scipyRoot with %s method' %method)
                    sol = scipyRoot(greensIteration_FixedPoint,psiIn, args=gi_args, method=method, options=solverOptions)
                    print(sol.success)
                    print(sol.message)
                    psiOut = sol.x
                    Done = True
                except Exception:
                    if np.abs(eigenvalueDiff) < tol/10:
                        print("Rootfinding didn't converge but eigenvalue is converged.  Exiting because this is probably due to degeneracy in the space.")
    #                         targets = tree.extractPhi(m)
                        psiOut = np.append(orbitals[:,m], Energies['orbitalEnergies'][m])
                        Done=True
                    else:
                        print('Not converged.  What to do?')
                        return
            orbitals[:,m] = np.copy(psiOut[:-1])
            Energies['orbitalEnergies'][m] = np.copy(psiOut[-1])
             
            print('Used %i iterations for wavefunction %i' %(greenIterationsCount,m))
            
    
        
        ## Sort by eigenvalue
        orbitals, Energies['orbitalEnergies'] = sortByEigenvalue(orbitals,Energies['orbitalEnergies'])
        
        fermiObjectiveFunction = fermiObjectiveFunctionClosure(Energies,nElectrons)        
        eF = brentq(fermiObjectiveFunction, Energies['orbitalEnergies'][0], 1, xtol=1e-14)
        print('Fermi energy: ', eF)
        exponentialArg = (Energies['orbitalEnergies']-eF)/Sigma
        occupations = 2*1/(1+np.exp( exponentialArg ) )  # these are # of electrons, not fractional occupancy.  Hence the 2*
    
    #         occupations = computeOccupations(Energies['orbitalEnergies'], nElectrons, Temperature)
        print('Occupations: ', occupations)
        Energies['Eband'] = np.sum( (Energies['orbitalEnergies']-Energies['gaugeShift']) * occupations)
    
    
        print()  
        print()
     
    
        
    
        
    
        oldDensity = np.copy(RHO)
        
        RHO = np.zeros(nPoints)
        for m in range(nOrbitals):
            RHO += orbitals[:,m]**2 * occupations[m]
        newDensity = np.copy(RHO)
        
    
        if SCFcount==1: # not okay anymore because output density gets reset when tolerances get reset.
            outputDensities[:,0] = np.copy(newDensity)
#             scf_args['outputDensities']=outputDensities
        else:
            
    #             outputDensities = np.concatenate( ( outputDensities, np.reshape(np.copy(newDensity), (nPoints,1)) ), axis=1)
            
            if (SCFcount-1)<mixingHistoryCutoff:
                outputDensities = np.concatenate( (outputDensities, np.reshape(np.copy(newDensity), (nPoints,1))), axis=1)
                print('Concatenated outputDensity.  Now has shape: ', np.shape(outputDensities))
            else:
                print('Beyond mixingHistoryCutoff.  Replacing column ', (SCFcount-1)%mixingHistoryCutoff)
    #                                 print('Shape of oldOrbitals[:,m]: ', np.shape(oldOrbitals[:,m]))
                outputDensities[:,(SCFcount-1)%mixingHistoryCutoff] = newDensity
        
    #         print('Sample of output densities:')
    #         print(outputDensities[0,:])    
        integratedDensity = np.sum( newDensity*W )
        densityResidual = np.sqrt( np.sum( (newDensity-oldDensity)**2*W ) )
        print('Integrated density: ', integratedDensity)
        print('Density Residual ', densityResidual)
        
        
            
    
    
        
        
        Energies['Eband'] = np.sum( (Energies['orbitalEnergies']-Energies['gaugeShift']) * occupations)
        Energies['Etotal'] = Energies['Eband'] - Energies['Ehartree'] + Energies['Ex'] + Energies['Ec'] - Energies['Vx'] - Energies['Vc'] + Energies['Enuclear']
        
    
        for m in range(nOrbitals):
            print('Orbital %i error: %1.3e' %(m, Energies['orbitalEnergies'][m]-referenceEigenvalues[m]-Energies['gaugeShift']))
        
        
        energyResidual = abs( Energies['Etotal'] - Energies['Eold'] )  # Compute the energyResidual for determining convergence
        Energies['Eold'] = np.copy(Energies['Etotal'])
        
        
        
        """
        Print results from current iteration
        """
    
        print('Orbital Energies: ', Energies['orbitalEnergies']) 
    
        print('Updated V_x:                           %.10f Hartree' %Energies['Vx'])
        print('Updated V_c:                           %.10f Hartree' %Energies['Vc'])
        
        print('Updated Band Energy:                   %.10f H, %.10e H' %(Energies['Eband'], Energies['Eband']-referenceEnergies['Eband']) )
    #         print('Updated Kinetic Energy:                 %.10f H, %.10e H' %(Energies['kinetic'], Energies['kinetic']-Ekinetic) )
        print('Updated E_Hartree:                      %.10f H, %.10e H' %(Energies['Ehartree'], Energies['Ehartree']-referenceEnergies['Ehartree']) )
        print('Updated E_x:                           %.10f H, %.10e H' %(Energies['Ex'], Energies['Ex']-referenceEnergies['Eexchange']) )
        print('Updated E_c:                           %.10f H, %.10e H' %(Energies['Ec'], Energies['Ec']-referenceEnergies['Ecorrelation']) )
    #         print('Updated totalElectrostatic:            %.10f H, %.10e H' %(tree.totalElectrostatic, tree.totalElectrostatic-Eelectrostatic))
        print('Total Energy:                          %.10f H, %.10e H' %(Energies['Etotal'], Energies['Etotal']-referenceEnergies['Etotal']))
        print('Energy Residual:                        %.3e' %energyResidual)
        print('Density Residual:                       %.3e\n\n'%densityResidual)
    
        scf_args['energyResidual']=energyResidual
        scf_args['densityResidual']=densityResidual
            
    #         if vtkExport != False:
    #             filename = vtkExport + '/mesh%03d'%(SCFcount-1) + '.vtk'
    #             Energies['Etotal']xportGridpoints(filename)
    
        printEachIteration=True
    
        if printEachIteration==True:
            header = ['Iteration', 'densityResidual', 'orbitalEnergies','bandEnergy', 'kineticEnergy', 
                      'exchangeEnergy', 'correlationEnergy', 'hartreeEnergy', 'totalEnergy']
        
            myData = [SCFcount, densityResidual, Energies['orbitalEnergies'], Energies['Eband'], Energies['kinetic'], 
                      Energies['Ex'], Energies['Ec'], Energies['Ehartree'], Energies['Etotal']]
            
        
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
        scf_args['Veff']=Veff
    
    
        return newDensity-oldDensity
    return scfFixedPoint, scf_args



