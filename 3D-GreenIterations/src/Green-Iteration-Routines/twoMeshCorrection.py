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
import BaryTreeInterface as BT
# from orthogonalizationRoutines import modifiedGramSchmidt_singleOrbital_transpose as mgs
from greenIterationFixedPoint import greensIteration_FixedPoint_Closure
import moveData_wrapper as MOVEDATA



 
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
#         appendedWeights = np.append(np.zeros_like(W), 10.0)   # NOTE: The appended weight was previously set to 10, giving extra weight to the eigenvalue 
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
        newOrbitals[m,:] = orbitals[newOrder[m],:]            
   
    return newOrbitals, orbitalEnergies
      
def twoMeshCorrectionClosure(scf_args): 
    
    def twoMeshCorrection(RHO,scf_args, abortAfterInitialHartree=False):
        
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
        Vext_local=np.copy(scf_args['Vext_local'])
        Vext_local_fine=np.copy(scf_args['Vext_local_fine'])
        Veff_local_old=np.copy(scf_args['Veff_local'])
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

        twoMesh=True
        
            
            
        SCFindex = SCFcount
        if SCFcount>TwoMeshStart:
            SCFindex = SCFcount - TwoMeshStart
            

        print("Interpolating density from %i to %i point mesh." %(len(X),len(Xf)))
        numberOfCells=len(pointsPerCell_coarse)
        RHOf = interpolation_wrapper.callInterpolator(X,  Y,  Z,  RHO, pointsPerCell_coarse,
                                                           Xf, Yf, Zf, pointsPerCell_fine, 
                                                           numberOfCells, order, GPUpresent)
            

           
          
        rprint(rank,"Using singularity subtraction in Hartree solve.")
        kernelName = "coulomb"
        numberOfKernelParameters=1
        kernelParameters=np.array([gaussianAlpha])
         
        
        start = MPI.Wtime()
        
        
#             print("Rank %i calling treecode through wrapper..." %(rank))
        
        treecode_verbosity=0
        
        numSources = len(Xf)
        sourceX=Xf
        sourceY=Yf
        sourceZ=Zf
        sourceRHO=RHOf
        sourceW=Wf

            
#             singularityHandling="skipping"
#             print("Forcing the Hartree solve to use singularity skipping.")

        rprint(rank,"Performing Hartree solve on %i mesh points" %numSources)
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
        V_hartreeNew = BT.callTreedriver(  
                                            nPoints, numSources, 
                                            np.copy(X), np.copy(Y), np.copy(Z), np.copy(RHO), 
                                            np.copy(sourceX), np.copy(sourceY), np.copy(sourceZ), np.copy(sourceRHO), np.copy(sourceW),
                                            kernel, numberOfKernelParameters, kernelParameters, 
                                            singularity, approximation, computeType,
                                            treecodeOrder, theta, maxParNode, batchSize,
                                            GPUpresent, treecode_verbosity, sizeCheck=1.0
                                            )
        

        rprint(rank,'Convolution time: ', MPI.Wtime()-start)
            

        
        """ 
        Compute the new orbital and total energies 
        """
        
        ## Energy update after computing Vhartree
          
        comm.barrier()    
        Energies['Ehartree'] = 1/2*global_dot(W, RHO * V_hartreeNew, comm)
        exchangeOutput = exchangeFunctional.compute(RHO)
        correlationOutput = correlationFunctional.compute(RHO)        
        Energies['Ex'] = global_dot( W, RHO * np.reshape(exchangeOutput['zk'],np.shape(RHO)), comm )
        Energies['Ec'] = global_dot( W, RHO * np.reshape(correlationOutput['zk'],np.shape(RHO)), comm )
        
        Vx = np.reshape(exchangeOutput['vrho'],np.shape(RHO))
        Vc = np.reshape(correlationOutput['vrho'],np.shape(RHO))
        
#         Energies['Vx'] = np.sum(W * RHO * Vx)
#         Energies['Vc'] = np.sum(W * RHO * Vc)

        Energies['Vx'] = global_dot(W, RHO * Vx,comm)
        Energies['Vc'] = global_dot(W, RHO * Vc,comm)
        Energies["Repulsion"] = global_dot(RHO, Vext_local*W, comm)
        
        Veff_local_new = V_hartreeNew + Vx + Vc + Vext_local + gaugeShift
        
        

        ## Update each of the eigenvalues with new local and nonlocal pieces
        for m in range(nOrbitals): 
            
            # Extract the mth eigenpair
            eigenvalue=Energies['orbitalEnergies'][m]
            psi_coarse = orbitals[m,:]
            
            
            # Obtain wavefunction and local potential on the fine mesh
            numberOfCells=len(pointsPerCell_coarse)
            psi_fine = interpolation_wrapper.callInterpolator(X,  Y,  Z, psi_coarse, pointsPerCell_coarse,
                                                           Xf, Yf, Zf, pointsPerCell_fine, 
                                                           numberOfCells, order, GPUpresent)
            Veff_local_new -= Vext_local           
#             Veff_local_fine_old = interpolation_wrapper.callInterpolator(X,  Y,  Z,  Veff_local_old, pointsPerCell_coarse,
#                                                            Xf, Yf, Zf, pointsPerCell_fine, 
#                                                            numberOfCells, order, GPUpresent)
            Veff_local_fine_new = interpolation_wrapper.callInterpolator(X,  Y,  Z,  Veff_local_new, pointsPerCell_coarse,
                                                           Xf, Yf, Zf, pointsPerCell_fine, 
                                                           numberOfCells, order, GPUpresent)
                
            Veff_local_fine_new += Vext_local_fine       
            Veff_local_new += Vext_local
            
            
            # Update local piece
#             eigenvalue += global_dot(W*psi_coarse,psi_coarse*(Veff_local_new-Veff_local_old), comm)


            oldLocal=global_dot(W*psi_coarse,psi_coarse*(Veff_local_old), comm)
            newLocal=global_dot(Wf*psi_fine,psi_fine*(Veff_local_fine_new), comm)
#             newLocal=global_dot(W*psi_coarse,psi_coarse*(Veff_local_new), comm)
            
            
#             eigenvalue -= oldLocal
#             eigenvalue += newLocal
            rprint(rank,"\n\nold Local to be subtracted: ", oldLocal)
            rprint(rank,"new Local to be added:      ", newLocal)
            
            
            # Update nonlocal piece
            V_nl_psi_fine = np.zeros(len(Veff_local_fine_new))
            V_nl_psi_coarse = np.zeros(len(Veff_local_old))
            V_nl_psi_coarse_new = np.zeros(len(Veff_local_old))
            for atom in atoms:
                 
                V_nl_psi_coarse += atom.V_nonlocal_pseudopotential_times_psi_SingleMesh(psi_coarse,W,comm=comm)
                V_nl_psi_fine += atom.V_nonlocal_pseudopotential_times_psi_fine(psi_fine,Wf,comm=comm)
                
                V_nl_psi_coarse_new += atom.V_nonlocal_pseudopotential_times_psi_coarse(psi_coarse,W,psi_fine,Wf,comm=comm)
                 
            oldNonLocal = global_dot(W*psi_coarse,V_nl_psi_coarse, comm)
#             newNonLocal = global_dot(W*psi_coarse,V_nl_psi_coarse_new, comm)
            newNonLocal = global_dot(Wf*psi_fine,V_nl_psi_fine, comm)
            
#             eigenvalue-=oldNonLocal
#             eigenvalue+=newNonLocal
            
            rprint(rank,"\n\nold nonlocal to be subtracted: ", oldNonLocal)
            rprint(rank,"new nonlocal to be added:      ", newNonLocal)
            
            
            f_coarse = -2* ( psi_coarse*Veff_local_new + V_nl_psi_coarse_new )
            f_fine = -2* ( psi_fine*Veff_local_fine_new + V_nl_psi_fine )
            
            
            # Compute new wavefunction
            numSources = len(Xf)
            sourceX=Xf
            sourceY=Yf
            sourceZ=Zf
            sourceF=f_fine
            sourceW=Wf
            
            
            kernel = BT.Kernel.YUKAWA
            numberOfKernelParameters=1
            k = np.sqrt(-2*Energies['orbitalEnergies'][m])
            kernelParameters=np.array([k])
            singularity=BT.Singularity.SUBTRACTION
            
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

            psiNew /= (4*np.pi)
            comm.barrier()  
            
            
            # Update eigenvalue
            deltaE = -global_dot( psi_coarse*(Veff_local_new)*(psi_coarse-psiNew), W, comm )
            deltaE -= global_dot( V_nl_psi_coarse_new*(psi_coarse-psiNew), W, comm ) 
            normSqOfPsiNew = global_dot( psiNew**2, W, comm)
            rprint(rank,"normSqOfPsiNew = ", normSqOfPsiNew)
            deltaE /= (normSqOfPsiNew)
                         
            rprint(rank,"Wavefunction %i, delta E = %f" %(m,deltaE))
            
            
            # Replace updated eigenvalue in array
            Energies['orbitalEnergies'][m] += deltaE
#             Energies['orbitalEnergies'][m]=eigenvalue
            
        ## Sort by eigenvalue
        
#         orbitals, Energies['orbitalEnergies'] = sortByEigenvalue(orbitals,Energies['orbitalEnergies'])
        
        if GPUpresent: MOVEDATA.callRemoveVectorFromDevice(orbitals)
        orbitals, Energies['orbitalEnergies'] = sortByEigenvalue(orbitals,Energies['orbitalEnergies'])
        if GPUpresent: MOVEDATA.callCopyVectorToDevice(orbitals)
            
        
        fermiObjectiveFunction = fermiObjectiveFunctionClosure(Energies,nElectrons)        
        eF = brentq(fermiObjectiveFunction, Energies['orbitalEnergies'][0], 1, xtol=1e-14)
        if verbosity>0: rprint(rank,'Fermi energy: ', eF)
        exponentialArg = (Energies['orbitalEnergies']-eF)/Sigma
        occupations = 2*1/(1+np.exp( exponentialArg ) )  # these are # of electrons, not fractional occupancy.  Hence the 2*
    
    #         occupations = computeOccupations(Energies['orbitalEnergies'], nElectrons, Temperature)
        if verbosity>0: rprint(rank,'Occupations: ', occupations)
        Energies['Eband'] = np.sum( (Energies['orbitalEnergies']-Energies['gaugeShift']) * occupations)
    
     
    
        
    
        
    
        
#         Energies["Repulsion"] = global_dot(RHO, Vext_local*W, comm)  # this doesn't need updating.
        
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
    
        rprint(rank,'Orbital Energies: ', Energies['orbitalEnergies']) 
    
        rprint(rank,'Updated V_x:                           %.10f Hartree' %Energies['Vx'])
        rprint(rank,'Updated V_c:                           %.10f Hartree' %Energies['Vc'])
        
        rprint(rank,'Updated Band Energy:                   %.10f H, %.10e H' %(Energies['Eband'], Energies['Eband']-referenceEnergies['Eband']) )
        rprint(rank,'Updated E_Hartree:                      %.10f H, %.10e H' %(Energies['Ehartree'], Energies['Ehartree']-referenceEnergies['Ehartree']) )
        rprint(rank,'Updated E_x:                           %.10f H, %.10e H' %(Energies['Ex'], Energies['Ex']-referenceEnergies['Eexchange']) )
        rprint(rank,'Updated E_c:                           %.10f H, %.10e H' %(Energies['Ec'], Energies['Ec']-referenceEnergies['Ecorrelation']) )
        rprint(rank,'Updated totalElectrostatic:            %.10f H, %.10e H' %(Energies['totalElectrostatic'], Energies['totalElectrostatic']-referenceEnergies["Eelectrostatic"]))
        rprint(rank,'Total Energy:                          %.10f H, %.10e H' %(Energies['Etotal'], Energies['Etotal']-referenceEnergies['Etotal']))
        rprint(rank,'Energy Residual:                        %.3e' %energyResidual)
#         rprint(rank,'Density Residual:                       %.3e\n\n'%densityResidual)
    
        scf_args['energyResidual']=energyResidual
        scf_args['densityResidual']=0.0
        
        
            
    #         if vtkExport != False:
    #             filename = vtkExport + '/mesh%03d'%(SCFcount-1) + '.vtk'
    #             Energies['Etotal']xportGridpoints(filename)
    
        printEachIteration=True
    
        if printEachIteration==True:
            header = ['Iteration', 'densityResidual', 'orbitalEnergies','bandEnergy', 'kineticEnergy', 
                      'exchangeEnergy', 'correlationEnergy', 'hartreeEnergy', 'totalEnergy', 'GItolerance']
         
            myData = [SCFcount, 0.0, Energies['orbitalEnergies']-Energies['gaugeShift'], Energies['Eband'], Energies['kinetic'], 
                      Energies['Ex'], Energies['Ec'], Energies['Ehartree'], Energies['Etotal'], scf_args['currentGItolerance']]


          
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
                

            
        ## Pack up scf_args
        scf_args['outputDensities']=outputDensities
        scf_args['inputDensities']=inputDensities
        scf_args['SCFcount']=SCFcount
        scf_args['Energies']=Energies
        scf_args['Times']=Times
        scf_args['orbitals']=orbitals
        scf_args['oldOrbitals']=oldOrbitals
        scf_args['Veff_local']=Veff_local_new
    
    
        return
    return twoMeshCorrection, scf_args



