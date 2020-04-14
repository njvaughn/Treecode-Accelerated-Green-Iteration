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
from meshUtilities import interpolateBetweenTwoMeshes
import BaryTreeInterface as BT
import orthogonalization_wrapper as ORTH
import moveData_wrapper as MOVEDATA

# try:
#     import treecodeWrappers_distributed as treecodeWrappers
# except ImportError:
#     print('Unable to import treecodeWrapper due to ImportError')
# except OSError:
#     print('Unable to import treecodeWrapper due to OSError')
    
# from orthogonalizationRoutines import modifiedGramSchmidt_singleOrbital_transpose as mgs
# from orthogonalizationRoutines import mask

import interpolation_wrapper


def greensIteration_FixedPoint_Closure(gi_args):
#     gi_args_out = {}
    def greensIteration_FixedPoint(psiIn, gi_args):
        # what other things do we need?  Energies, Times, orbitals, Veff, runtime constants (symmetricIteration, GPUpresent, subtractSingularity, treecode, outputfiles, ...)  
        verbosity=1
        
#         print("entered greensIteration_FixedPoint.")
        
        ## UNPACK GIARGS
#         print('MEMORY USAGE: ', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss )
#         GPUtil.showUtilization()
        orbitals = gi_args['orbitals']
        oldOrbitals = gi_args['oldOrbitals']
        Energies = gi_args['Energies']
        Times = gi_args['Times']
        
        Veff_local = gi_args['Veff_local']
        Vext_local = gi_args['Vext_local']
        Vext_local_fine = gi_args['Vext_local_fine']
        
        
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
        Xf = gi_args['Xf']
        Yf = gi_args['Yf']
        Zf = gi_args['Zf']  
        Wf = gi_args['Wf']
        pointsPerCell_coarse = gi_args['pointsPerCell_coarse']
        pointsPerCell_fine = gi_args['pointsPerCell_fine']
#         print("X, Y, Z, W: ", X[:3], Y[:3], Z[:3], W[:3])
        gradientFree = gi_args['gradientFree']
        SCFcount = gi_args['SCFcount']
        greenIterationsCount = gi_args['greenIterationsCount']
        residuals = gi_args['residuals']
        eigenvalueResiduals = gi_args['eigenvalueResiduals']
        greenIterationOutFile = gi_args['greenIterationOutFile']
        referenceEigenvalues = gi_args['referenceEigenvalues']
        updateEigenvalue = gi_args['updateEigenvalue']
        coreRepresentation = gi_args['coreRepresentation']
        atoms=gi_args['atoms']
#         nearbyAtoms=gi_args["nearbyAtoms"]
        coarse_order=gi_args["order"]
        fine_order=gi_args["fine_order"]
        regularize=gi_args['regularize']
        epsilon=gi_args["epsilon"]
        TwoMeshStart=gi_args["TwoMeshStart"]
        singleWavefunctionOrthogonalization=gi_args["singleWavefunctionOrthogonalization"]
        
        order=coarse_order

        
#         print('Who called F(x)? ', inspect.stack()[2][3])
        inputWave = np.copy(psiIn[:-1])
        
#         TwoMeshStart=1
        
        if ( (len(X)!=len(Xf)) and (SCFcount>TwoMeshStart)  ):
            twoMesh=True
        else:
            twoMesh=False
    

        
        Times['totalIterationCount'] += 1
        
        
         
        oldOrbitals[m,:] = np.copy(psiIn[:-1])    
        orbitals[m,:] = np.copy(psiIn[:-1])
        Energies['orbitalEnergies'][m] = np.copy(psiIn[-1])
        
        # get the input wavefunction on the fine mesh.
        
        if ( twoMesh ):
            start=time.time()
#             interpolatedInputWavefunction = interpolateBetweenTwoMeshes(X, Y, Z, orbitals[m,:], coarse_order,
#                                                                Xf, Yf, Zf, fine_order) 
#             interpolatedInputWavefunction = interpolateBetweenTwoMeshes(X, Y, Z, orbitals[m,:], pointsPerCell_coarse,
#                                                             Xf, Yf, Zf, pointsPerCell_fine)
            
            numberOfCells=len(pointsPerCell_coarse)
            interpolatedInputWavefunction = interpolation_wrapper.callInterpolator(X,  Y,  Z,  orbitals[m,:], pointsPerCell_coarse,
                                                           Xf, Yf, Zf, pointsPerCell_fine, 
                                                           numberOfCells, order, GPUpresent)
            
#             ## Want to use interpolated Vlocal???
             
            # To obtain Veff_local_fine:
            # 1. subtract out the coarse Vext
            # 2. interpolate the remainder on to the fine mesh
            # 3. add in the fine Vext (from the radial data)
            # 4. add back in the subtracted piece on the coarse mesh.
             
            Veff_local -= Vext_local                                                                    # step 1          
#             Veff_local_fine = interpolateBetweenTwoMeshes(X, Y, Z, Veff_local, pointsPerCell_coarse,
#                                                             Xf, Yf, Zf, pointsPerCell_fine)             # step 2 
            Veff_local_fine = interpolation_wrapper.callInterpolator(X,  Y,  Z,  Veff_local, pointsPerCell_coarse,
                                                           Xf, Yf, Zf, pointsPerCell_fine, 
                                                           numberOfCells, order, GPUpresent)
              
            Veff_local_fine += Vext_local_fine                                                          # step 3
            Veff_local += Vext_local                                                                    # step 4
            end=time.time()
            
            
#             ## OR do I want to just interpolate the whole Veff to the fine grid?
#             Veff_local_fine = interpolateBetweenTwoMeshes(X, Y, Z, Veff_local, pointsPerCell_coarse,
#                                                             Xf, Yf, Zf, pointsPerCell_fine) 
            
            if verbosity>-1: rprint(rank,"Time to interpolate wavefunction: ", end-start) 
        else:
            interpolatedInputWavefunction=orbitals[m,:]
            Veff_local_fine=Veff_local
        
        
#         print("length of interpolated wavefunction: ", len(interpolatedInputWavefunction))
     
        if coreRepresentation=='AllElectron':
            f = -2*orbitals[m,:]*Veff_local
        elif coreRepresentation=='Pseudopotential': 
            start=time.time()
            V_nl_psi_fine = np.zeros(len(Veff_local_fine))
            V_nl_psi_coarse = np.zeros(len(Veff_local))
#             if verbosity>0: rprint(rank,"SKIPPING NONLOCAL POTENTIAL ::::::::::::::: FOR TESTING ONLY")
#             print(nearbyAtoms)
#             comm.barrier()
#             exit(-1)
#             for atom in nearbyAtoms:
            for atom in atoms:
                
#                 pass
#                 V_nl_psi += atom.V_nonlocal_pseudopotential_times_psi(orbitals[m,:],Wf,interpolatedPsi=interpolatedInputWavefunction,comm=comm)
#                 V_nl_psi_fine += atom.V_nonlocal_pseudopotential_times_psi(interpolatedInputWavefunction,Wf,interpolatedPsi=interpolatedInputWavefunction,comm=comm,outputMesh="fine")
#                 V_nl_psi_coarse += atom.V_nonlocal_pseudopotential_times_psi(orbitals[m,:],Wf,interpolatedPsi=interpolatedInputWavefunction,comm=comm,outputMesh="coarse")

                if twoMesh: 
                    V_nl_psi_coarse += atom.V_nonlocal_pseudopotential_times_psi_coarse(orbitals[m,:],W,interpolatedInputWavefunction,Wf,comm=comm)
                else:
                    V_nl_psi_coarse += atom.V_nonlocal_pseudopotential_times_psi_SingleMesh(orbitals[m,:],W,comm=comm)
                    
                if twoMesh: 
                    V_nl_psi_fine += atom.V_nonlocal_pseudopotential_times_psi_fine(interpolatedInputWavefunction,Wf,comm=comm)
                else:
                    pass

#                 norm_of_V_nl_psi = np.sqrt(global_dot(V_nl_psi**2,W,comm))
#                 if verbosity>0: rprint(rank,"NORM OF V_NL_PSI = ", norm_of_V_nl_psi)
#             f = -2* ( orbitals[m,:]*Veff_local + V_nl_psi )

#             if fine_order!=coarse_order:
#                 V_nl_psi_coarse = interpolateBetweenTwoMeshes(Xf, Yf, Zf, V_nl_psi_fine, fine_order,X, Y, Z, coarse_order) 
#             else:
#                 V_nl_psi_coarse=V_nl_psi_fine


            if twoMesh: f_fine = -2* ( interpolatedInputWavefunction*Veff_local_fine + V_nl_psi_fine )
            f_coarse = -2* ( orbitals[m,:]*Veff_local + V_nl_psi_coarse )
            end=time.time()
#             if verbosity>0: rprint(rank,"Constructing f with nonlocal routines took %f seconds." %(end-start))
        else:
            print("coreRepresentation not set to allowed value. Exiting from greenIterationFixedPoint.")
            return
        
        
#         print("ANY NaNs??? ", np.isnan(f).any())
        if twoMesh:
            if np.isnan(f_fine).any():
                print("NaNs detected in f = -2*orbitals[m,:]*Veff_local.  Exiting")
                exit(-1)
        oldEigenvalue =  Energies['orbitalEnergies'][m] 
        k = np.sqrt(-2*Energies['orbitalEnergies'][m])
#         if verbosity>0: rprint(rank, 'k = ', k)
    
        psiNew = np.zeros(nPoints)
        
        if Energies['orbitalEnergies'][m] < 10.25**100: 
            
            
            startTime=time.time()
#             singularityHandling="skipping"
#             if verbosity>0: rprint(rank,"singularityHandling = ", singularityHandling)
            if regularize==False:
#                 if verbosity>0: rprint(rank,"Using singularity subtraction kernel in Green Iteration.")
                kernelName = "yukawa"
                numberOfKernelParameters=1
                kernelParameters=np.array([k])
            elif regularize==True:
#                 if verbosity>0: rprint(rank,"Using regularized yukawa for Green Iteration with epsilon = ", epsilon)
                kernelName="regularized-yukawa"
                numberOfKernelParameters=2
                kernelParameters=np.array([k, epsilon])
                if epsilon!=0.0:
                    print("WARNING: SHOULD EPSILON BE NONZERO?")
                
                
            
#             kappa = k
            
            treecode_verbosity=0
            
            
            
            if twoMesh:  # idea: only turn on the two mesh if beyond 4 SCF iterations
                numSources = len(Xf)
                sourceX=Xf
                sourceY=Yf
                sourceZ=Zf
                sourceF=f_fine
                sourceW=Wf
            else: 
                numSources = len(X)
                sourceX=X
                sourceY=Y
                sourceZ=Z
                sourceF=f_coarse
                sourceW=W
            
            
#             for batchSize in [1000, 2000, 4000, 8000, 16000]:
#                 for maxParNode in [1000, 2000, 4000, 8000, 16000]:
            kernel = BT.Kernel.YUKAWA
            if singularityHandling=="subtraction":
                singularity=BT.Singularity.SUBTRACTION
            elif singularityHandling=="skipping":
                singularity=BT.Singularity.SKIPPING
            else:
                print("What should singularityHandling be?")
                exit(-1)
            
            if approximationName=="lagrange":
                approximation=BT.Approximation.LAGRANGE
            elif approximationName=="hermite":
                approximation=BT.Approximation.HERMITE
            else:
                print("What should approximationName be?")
                exit(-1)
            
            computeType=BT.ComputeType.PARTICLE_CLUSTER
            
              
            comm.barrier()
            startTime = time.time()
            psiNew = BT.callTreedriver(
                                        nPoints, numSources, 
                                        np.copy(X), np.copy(Y), np.copy(Z), np.copy(f_coarse), 
                                        np.copy(sourceX), np.copy(sourceY), np.copy(sourceZ), np.copy(sourceF), np.copy(sourceW),
                                        kernel, numberOfKernelParameters, kernelParameters, 
                                        singularity, approximation, computeType,
                                        treecodeOrder, theta, maxParNode, batchSize,
                                        GPUpresent, treecode_verbosity, sizeCheck=1.0
                                        )

            if singularityHandling=="skipping": psiNew /= (4*np.pi)
            if singularityHandling=="subtraction": psiNew /= (4*np.pi)
#             if singularityHandling=="regularized": psiNew /= (4*np.pi)

            comm.barrier()
            convolutionTime = time.time()-startTime
            if verbosity>0: rprint(rank,'Convolution time: ', convolutionTime)
            Times['timePerConvolution'] = convolutionTime
            if verbosity>0: rprint(rank,"Batch size %i, cluster size %i, time per convolution %f" %(batchSize,maxParNode,convolutionTime))
            
#             rprint(rank,"Exiting because only interested in time per convolution.")
#             exit(-1)

        else:
            print('Exiting because energy too close to 0')
            exit(-1)
#             gpuHelmholtzConvolution[blocksPerGrid, threadsPerBlock](np.array([X,Y,Z,f,W]),np.array([X,Y,Z,f,W]),psiNew,k)
        
        
#         """ Apply MASK """
#         print("=================APPLYING MASK===================")
#         print("=================APPLYING MASK===================")
#         print("=================APPLYING MASK===================")
#         domainSize=np.max(X)
#         psiNew = mask(psiNew,X,Y,Z,domainSize)

        
        """ Method where you dont compute kinetics, from Harrison """
        if updateEigenvalue==True:
            # update the energy first
            
            orthWavefunction2 = np.zeros(nPoints)
            if ( (gradientFree==True) and (SCFcount>-1)):                 
                
                psiNewNorm = np.sqrt( global_dot( psiNew, psiNew*W, comm))
                if verbosity>0: rprint(rank,"psiNewNorm = %f" %psiNewNorm)
                
        
                deltaE = -global_dot( orbitals[m,:]*(Veff_local)*(orbitals[m,:]-psiNew), W, comm )
                
                if coreRepresentation=="AllElectron":
                    pass
                elif coreRepresentation=="Pseudopotential":
                    
                    
                    
                    
                    ## DONT NEED TO INTERPOLATE AGAIN.  interpolatedInputWavefunction IS THE SAME AS BEFORE.
                    # Obtain old orbital on the same mesh as the projectors
#                     if fine_order!=coarse_order:
#                         start=time.time()
#                         interpolatedInputWavefunction = interpolateBetweenTwoMeshes(X, Y, Z, orbitals[m,:], coarse_order,
#                                                                            Xf, Yf, Zf, fine_order) 
#                         end=time.time()
#                         if verbosity>0: rprint(rank,"Time to interpolate wavefunction for eigenvalue update: ", end-start) 
#                     else:
#                         interpolatedInputWavefunction=orbitals[m,:]
#                     
                    
        
        
#                     if verbosity>0: rprint(rank,"SKIPPING NONLOCAL POTENTIAL ::::::::::::::: FOR TESTING ONLY")

                    ## Actually, there is no need to recompute V_nl_psi, it is the same as before.
#                     # Compute action of V_nl on the old orbital
#                     V_nl_psi = np.zeros(nPoints)
#                     for atom in atoms:
#                         V_nl_psi+= atom.V_nonlocal_pseudopotential_times_psi(orbitals[m,:],Wf,interpolatedPsi=interpolatedInputWavefunction,comm=comm)

#                     pass
#                     # Compute the delta E, here using the difference between the new psi and old psi
                    deltaE -= global_dot( V_nl_psi_coarse*(orbitals[m,:]-psiNew), W, comm ) 
                else: 
                    print("Invalid coreRepresentation.")
                    exit(-1)
                normSqOfPsiNew = global_dot( psiNew**2, W, comm)
                deltaE /= (normSqOfPsiNew)  # divide by norm squared, according to Harrison-Fann- et al

#                 deltaE/=2 # do a simple mixing on epsilon, help with oscillations.
#                 if verbosity>0: rprint(rank,"Halving the deltaE to try to help with oscillations.")
                
                if verbosity>0: rprint(rank,"deltaE = %f" %deltaE)
                Energies['orbitalEnergies'][m] += deltaE
                if verbosity>0: rprint(rank,"Energies['orbitalEnergies'][m] = %f" %Energies['orbitalEnergies'][m])
                orbitals[m,:] = np.copy(psiNew)
                
        
                n,M = np.shape(orbitals) 
                if singleWavefunctionOrthogonalization==True:
#                     start=time.time()
#                     orthWavefunction = mgs(orbitals,W,m, comm)
#                     end=time.time()
#                     rprint(rank,"Original orthogonalizing wavefunctiong %i took %f seconds " %(m, end-start))
#                     
                    start=time.time()
                    U=np.copy(orbitals[m])
                    if GPUpresent: MOVEDATA.callCopyVectorToDevice(U)
#                     if GPUpresent: MOVEDATA.callCopyVectorToDevice(orbitals)
                    ORTH.callOrthogonalization(orbitals, U, W, m, GPUpresent)
                    if GPUpresent: MOVEDATA.callCopyVectorFromDevice(U)
                    orthWavefunction=np.copy(U)
#                     orbitals[m]=np.copy(U)
#                     if GPUpresent: MOVEDATA.callRemoveVectorFromDevice(orbitals)
        
#                     orthWavefunction = mgs(orbitals,W,m, comm)
                    end=time.time()
                    rprint(rank,"New orthogonalizing wavefunctiong %i took %f seconds " %(m, end-start))
                    orbitals[m,:] = np.copy(orthWavefunction)

        
                if greenIterationsCount==1:
                    eigenvalueHistory = np.array(Energies['orbitalEnergies'][m])
                else:
                    eigenvalueHistory = gi_args['eigenvalueHistory']
                    eigenvalueHistory = np.append(eigenvalueHistory, Energies['orbitalEnergies'][m])
                if verbosity>0: rprint(rank,'eigenvalueHistory: \n',eigenvalueHistory)
                
                
                 
        
            elif ( (gradientFree==False) or (gradientFree=='Laplacian') ):
                if verbosity>0: rprint(rank,'gradient and laplacian methods not set up for tree-free')
                return
                
            else:
                if verbosity>0: rprint(rank,'Not updating eigenvalue.  Is that intended?')
                return
            
        else:  # Explicitly choosing to not update Eigenvalue.  Still orthogonalize
            print("Not updating eigenvalue because updateEigenvalue!=True")
            exit(-1)
#             orbitals[m,:] = np.copy(psiNew)
#             n,M = np.shape(orbitals) 
#             orthWavefunction = mgs(orbitals,W,m,comm)
#             orbitals[m,:] = np.copy(orthWavefunction) 
#             if greenIterationsCount==1:
#                 eigenvalueHistory = np.array(Energies['orbitalEnergies'][m])
#             else:
#                 eigenvalueHistory = gi_args['eigenvalueHistory']
#                 eigenvalueHistory = np.append(eigenvalueHistory, Energies['orbitalEnergies'][m])
#             if rank==0: print('eigenvalueHistory (should be constant): \n',eigenvalueHistory)
#             gi_args['eigenvalueDiff']=0
#             deltaE=0
#             gi_args['eigenvalueHistory']=eigenvalueHistory
            
    
        
        if Energies['orbitalEnergies'][m]>0.0:
#         if Energies['orbitalEnergies'][m]>Energies['gaugeShift']:
#             Energies['orbitalEnergies'][m] = Energies['gaugeShift'] - np.random.randint(10)
            rand = np.random.rand(1)
#             Energies['orbitalEnergies'][m] = -2
#             Energies['orbitalEnergies'][m] = Energies['gaugeShift'] - 3*rand
            Energies['orbitalEnergies'][m] = Energies['gaugeShift']
#             if verbosity>0: rprint(rank,'Energy eigenvalue was positive, setting to  ',-2 )
#             if verbosity>0: rprint(rank,'Energy eigenvalue was positive, setting to gauge shift - ',( 3*rand) )
            if verbosity>0: rprint(rank,'Energy eigenvalue was positive, setting to gauge shift' )
            
            # use whatever random shift the root computed.
            Energies['orbitalEnergies'][m] = comm.bcast(Energies['orbitalEnergies'][m], root=0)
            
            if greenIterationsCount%10==0:
                # Positive energy after 10 iterations..., scramble wavefunction and restart.
                orbitals[m,:] = np.ones(len(np.copy(orbitals[m,:])))
            
        
    #     tree.printWavefunctionNearEachAtom(m)
            
    #     residualVector = orbitals[m,:] - oldOrbitals[m,:]
        psiOut = np.append(np.copy(orbitals[m,:]), np.copy(Energies['orbitalEnergies'][m]))
        residualVector = psiOut - psiIn
        
        
        
#         print('Max value of wavefunction: ', np.max(np.abs(orbitals[m,:])))
        loc = np.argmax(np.abs(residualVector[:-1]))
#         print('Largest residual: ', residualVector[loc])
#         print('Value at that point: ', psiOut[loc])
#         print('Location of max residual: ', X[loc], Y[loc], Z[loc])
    #     residualVector = -(psiIn - orbitals[m,:])
    
        newEigenvalue = Energies['orbitalEnergies'][m]
        
        
            
    
        
        if symmetricIteration==False:
#             normDiff = np.sqrt( np.sum( (orbitals[m,:]-oldOrbitals[m,:])**2*W ) )
            normDiff = np.sqrt( global_dot( (orbitals[m,:]-oldOrbitals[m,:])**2,W, comm ) )
        elif symmetricIteration==True:
#             normDiff = np.sqrt( np.sum( (orbitals[m,:]*sqrtV-oldOrbitals[m,:]*sqrtV)**2*W ) )
            normDiff = np.sqrt( global_dot( (orbitals[m,:]*sqrtV-oldOrbitals[m,:]*sqrtV)**2,W,comm ) )
        eigenvalueDiff = abs(newEigenvalue - oldEigenvalue)    
        
        if rank==0:
            residuals[m] = normDiff
        orbitalResidual = np.copy(normDiff)
        
        eigenvalueResiduals[m]=eigenvalueDiff
        
        
    
        if verbosity>0: rprint(rank,'Orbital %i error and eigenvalue residual:   %1.3e and %1.3e' %(m,Energies['orbitalEnergies'][m]-referenceEigenvalues[m]-Energies['gaugeShift'], eigenvalueDiff))
        if verbosity>0: rprint(rank,'Orbital %i wavefunction residual: %1.3e\n\n' %(m, orbitalResidual))
    
    
    
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
        
        gi_args['eigenvalueResiduals']=eigenvalueResiduals
        
        
        
        return residualVector
    
    return greensIteration_FixedPoint, gi_args 