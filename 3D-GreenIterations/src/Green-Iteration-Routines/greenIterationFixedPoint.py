import inspect
import numpy as np
import time
import csv
import resource
# import GPUtil
import os
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

from mpiUtilities import global_dot, rprint
try:
    import treecodeWrappers_distributed as treecodeWrappers
except ImportError:
    print('Unable to import treecodeWrapper due to ImportError')
except OSError:
    print('Unable to import treecodeWrapper due to OSError')
    
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
        Veff_local = gi_args['Veff_local']
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
        referenceEigenvalues = gi_args['referenceEigenvalues']
        updateEigenvalue = gi_args['updateEigenvalue']
        coreRepresentation = gi_args['coreRepresentation']
        atoms=gi_args['atoms']

        
#         print('Who called F(x)? ', inspect.stack()[2][3])
        inputWave = np.copy(psiIn[:-1])
    

        
        Times['totalIterationCount'] += 1
        
        
         
        oldOrbitals[:,m] = np.copy(psiIn[:-1])    
        orbitals[:,m] = np.copy(psiIn[:-1])
        Energies['orbitalEnergies'][m] = np.copy(psiIn[-1])
        
    
     
        if coreRepresentation=='AllElectron':
            f = -2*orbitals[:,m]*Veff_local
        elif coreRepresentation=='Pseudopotential': 
            V_nl_psi = np.zeros(nPoints)
            for atom in atoms:
                V_nl_psi += atom.V_nonlocal_pseudopotential_times_psi(X,Y,Z,orbitals[:,m],W,comm)
            f = -2* ( orbitals[:,m]*Veff_local + V_nl_psi )
            rprint(rank,"Constructed f with nonlocal routines.")
        else:
            print("coreRepresentation not set to allowed value. Exiting from greenIterationFixedPoint.")
            return
        
        
        oldEigenvalue =  Energies['orbitalEnergies'][m] 
        k = np.sqrt(-2*Energies['orbitalEnergies'][m])
#         print('k = ', k)
    
        phiNew = np.zeros(nPoints)
        
        if Energies['orbitalEnergies'][m] < 10.25**100: 
            
            
            startTime=time.time()
            kernelName = "yukawa"
            kappa = k
            startTime = time.time()
            comm.barrier()
            verbosity=0
            phiNew = treecodeWrappers.callTreedriver(nPoints, nPoints, 
                                                           np.copy(X), np.copy(Y), np.copy(Z), np.copy(f), 
                                                           np.copy(X), np.copy(Y), np.copy(Z), np.copy(f), np.copy(W),
                                                           kernelName, kappa, singularityHandling, approximationName, treecodeOrder, theta, maxParNode, batchSize, GPUpresent,verbosity)

            if singularityHandling=="skipping": phiNew /= (4*np.pi)
            if singularityHandling=="subtraction": phiNew /= (4*np.pi)

            convolutionTime = time.time()-startTime
            rprint(rank,'Using asymmetric singularity subtraction.  Convolution time: ', convolutionTime)
            comm.barrier()

        else:
            print('Exiting because energy too close to 0')
            exit(-1)
#             gpuHelmholtzConvolution[blocksPerGrid, threadsPerBlock](np.array([X,Y,Z,f,W]),np.array([X,Y,Z,f,W]),phiNew,k)
        

        
        """ Method where you dont compute kinetics, from Harrison """
        if updateEigenvalue==True:
            # update the energy first
            
            orthWavefunction2 = np.zeros(nPoints)
            if ( (gradientFree==True) and (SCFcount>-1)):                 
                
                psiNewNorm = np.sqrt( global_dot( phiNew, phiNew*W, comm))
                rprint(rank,"psiNewNorm = %f" %psiNewNorm)
                
        
                deltaE = -global_dot( orbitals[:,m]*(Veff_local)*(orbitals[:,m]-phiNew), W, comm )
                
                if coreRepresentation=="AllElectron":
                    pass
                elif coreRepresentation=="Pseudopotential":
#                     rprint(rank,"Need to address gradient-free eigenvalue update in Pseudopotential case.")
                    V_nl_psiDiff = np.zeros(nPoints)
                    for atom in atoms:
                        V_nl_psi+= atom.V_nonlocal_pseudopotential_times_psi(X,Y,Z,orbitals[:,m],W,comm)
#                         V_nl_psiDiff += atom.V_nonlocal_pseudopotential_times_psi(X,Y,Z,orbitals[:,m]-phiNew,W,comm)
                    deltaE -= global_dot( V_nl_psi*(orbitals[:,m]-phiNew), W, comm ) 
#                     deltaE -= global_dot( orbitals[:,m]* V_nl_psiDiff, W, comm ) 
                else: 
                    print("Invalid coreRepresentation.")
                    exit(-1)
                normSqOfPsiNew = global_dot( phiNew**2, W, comm)
                deltaE /= (normSqOfPsiNew)  # divide by norm squared, according to Harrison-Fann- et al

                deltaE/=2 # do a simple mixing on epsilon, help with oscillations.
                rprint(rank,"Halving the deltaE to try to help with oscillations.")
                
                rprint(rank,"deltaE = %f" %deltaE)
                Energies['orbitalEnergies'][m] += deltaE
                rprint(rank,"Energies['orbitalEnergies'][m] = %f" %Energies['orbitalEnergies'][m])
                orbitals[:,m] = np.copy(phiNew)
                
        
                n,M = np.shape(orbitals) 

                orthWavefunction = mgs(orbitals,W,m, n, M, comm)
                
                orbitals[:,m] = np.copy(orthWavefunction)

        
                if greenIterationsCount==1:
                    eigenvalueHistory = np.array(Energies['orbitalEnergies'][m])
                else:
                    eigenvalueHistory = gi_args['eigenvalueHistory']
                    eigenvalueHistory = np.append(eigenvalueHistory, Energies['orbitalEnergies'][m])
                rprint(rank,'eigenvalueHistory: \n',eigenvalueHistory)
                
                
                 
        
            elif ( (gradientFree==False) or (gradientFree=='Laplacian') ):
                rprint(rank,'gradient and laplacian methods not set up for tree-free')
                return
                
            else:
                rprint(rank,'Not updating eigenvalue.  Is that intended?')
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
#             Energies['orbitalEnergies'][m] = -2
            Energies['orbitalEnergies'][m] = Energies['gaugeShift'] - 3*rand
#             rprint(rank,'Energy eigenvalue was positive, setting to  ',-2 )
            rprint(rank,'Energy eigenvalue was positive, setting to gauge shift - ',( 3*rand) )
            
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
        
        
    
        rprint(rank,'Orbital %i error and eigenvalue residual:   %1.3e and %1.3e' %(m,Energies['orbitalEnergies'][m]-referenceEigenvalues[m]-Energies['gaugeShift'], eigenvalueDiff))
        rprint(rank,'Orbital %i wavefunction residual: %1.3e\n\n' %(m, orbitalResidual))
    
    
    
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