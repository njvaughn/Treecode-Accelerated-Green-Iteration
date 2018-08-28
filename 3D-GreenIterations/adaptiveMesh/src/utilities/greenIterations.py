"""
A CUDA version of the convolution for Green Iterations.
It is written so that each thread is responsible for one of the N target gridpoints.  
Each thread interacts with all M source midpoints. 
Note: the current implementation does not use shared memory, which would provide additional
speedup.  -- 03/19/2018 NV 
"""
# from numba import cuda
from math import sqrt,exp,factorial,pi
import math
import numpy as np
import os
import shutil
import csv
from timeit import default_timer as timer
from numpy import float32, float64

from hydrogenAtom import trueEnergy, trueWavefunction
# from convolution import gpuPoissonConvolution,gpuHelmholtzConvolutionSubractSingularity, cpuHelmholtzSingularitySubtract,cpuHelmholtzSingularitySubtract_allNumerical
from convolution import *


def modifiedGramSchrmidt(V,weights):
    n,k = np.shape(V)
    U = np.zeros_like(V)
    U[:,0] = V[:,0] / np.dot(V[:,0],V[:,0]*weights)
    for i in range(1,k):
        U[:,i] = V[:,i]
        for j in range(i):
#             print('Orthogonalizing %i against %i' %(i,j))
            U[:,i] -= (np.dot(U[:,i],U[:,j]*weights) / np.dot(U[:,j],U[:,j]*weights))*U[:,j]
        U[:i] /= np.dot(U[:,i],U[:,i]*weights)
        
    return U

def modifiedGramSchrmidt_noNormalization(V,weights):
    n,k = np.shape(V)
    U = np.zeros_like(V)
    U[:,0] = V[:,0] 
    for i in range(1,k):
        U[:,i] = V[:,i]
        for j in range(i):
            print('Orthogonalizing %i against %i' %(i,j))
            U[:,i] -= (np.dot(U[:,i],U[:,j]*weights) / np.dot(U[:,j],U[:,j]*weights))*U[:,j]
#         U[:i] /= np.dot(U[:,i],U[:,i]*weights)
        
    return U

def greenIterations_KohnSham_SCF(tree, intraScfTolerance, interScfTolerance, numberOfTargets, 
                                subtractSingularity, smoothingN, smoothingEps, inputFile='',iterationOutFile='iterationConvergence.csv',
                                onTheFlyRefinement = False, vtkExport=False, outputErrors=False): 
    '''
    Green Iterations for Kohn-Sham DFT using Clenshaw-Curtis quadrature.
    '''
    threadsPerBlock = 512
    blocksPerGrid = (numberOfTargets + (threadsPerBlock - 1)) // threadsPerBlock  # compute the number of blocks based on N and threadsPerBlock
    
    print('\nEntering greenIterations_KohnSham_SCF()')
    print('\nNumber of targets:   ', numberOfTargets)
    print('Threads per block:   ', threadsPerBlock)
    print('Blocks per grid:     ', blocksPerGrid)
    
    greenIterationCounter=1                                     # initialize the counter to counter the number of iterations required for convergence
    energyResidual = 1                                    # initialize the energyResidual to something that fails the convergence tolerance
    Eold = -0.5 + tree.gaugeShift

#     [Etrue, ExTrue, EcTrue, Eband] = np.genfromtxt(inputFile,dtype=[(str,str,int,int,float,float,float,float,float)])[4:8]
    [Etrue, ExTrue, EcTrue, Eband] = np.genfromtxt(inputFile)[4:8]
    print([Etrue, ExTrue, EcTrue, Eband])

    ### COMPUTE THE INITIAL HAMILTONIAN ###
    targets = tree.extractLeavesDensity()  
    sources = tree.extractLeavesDensity() 

    V_coulombNew = np.zeros((len(targets)))
    gpuPoissonConvolution[blocksPerGrid, threadsPerBlock](targets,sources,V_coulombNew)  # call the GPU convolution 
    tree.importVcoulombOnLeaves(V_coulombNew)
    tree.updateVxcAndVeffAtQuadpoints()
    tree.updateOrbitalEnergies()
    
    
    if vtkExport != False:
        filename = vtkExport + '/mesh%03d'%(greenIterationCounter-1) + '.vtk'
        tree.exportMeshVTK(filename)
        
    
    oldOrbitalEnergies = 10
    while ( energyResidual > interScfTolerance ):
        
        print('\nSCF Count ', greenIterationCounter)

        orbitalResidual = 10
        eigensolveCount = 0
        max_scfCount = 15
        while ( ( orbitalResidual > intraScfTolerance ) and ( eigensolveCount < max_scfCount) ):
            
            orbitalResidual = 0.0
            
            """ N ORBITAL HELMHOLTZ SOLVES """
            orbitals = np.zeros((len(targets),tree.nOrbitals))
            oldOrbitals = np.zeros((len(targets),tree.nOrbitals))
            for m in range(tree.nOrbitals):
                sources = tree.extractPhi(m)
                targets = np.copy(sources)
                weights = np.copy(targets[:,5])
                
                oldOrbitals[:,m] = np.copy(targets[:,3])
                if ( (m==1) and (greenIterationCounter<-1)):
                    print('Not computing new phi1')
                    orbitals[:,m] = np.copy(targets[:,3])
                else:
#                     weights = np.copy(targets[:,5])
    #                 phiNew = np.zeros((len(targets)))
                    if tree.orbitalEnergies[m] < tree.gaugeShift:
                        k = np.sqrt(-2*tree.orbitalEnergies[m])
                    else:
                        temporaryEpsilon = tree.gaugeShift-1/(m+1)
                        k = np.sqrt(-2*temporaryEpsilon)
                        print('Orbital %i energy %1.3e > Gauge Shift. Resetting to %1.3f' 
                              %(m,tree.orbitalEnergies[m],temporaryEpsilon))
                    
    
#                     phiT=np.copy(targets[:,3])
#                     vT = np.copy(targets[:,4])
#                     analyticPiece = -2*phiT*vT/k**2
#                      
#                  
#     # #                 print('max abs of analytic piece: ', max(abs(analyticPiece)))
#                     minIdx = np.argmin(analyticPiece)  
#                     maxIdx = np.argmax(analyticPiece) 
#                     print('~'*50)
#                     print('Orbital %i, k=%1.3f' %(m,k))
#                     print('1/(r*epsilon) near nucleus: ', 1/tree.rmin/tree.orbitalEnergies[m])
#                     print('min analytic piece: ', analyticPiece[minIdx])
#                     rmin = np.sqrt(targets[minIdx,0]**2 + targets[minIdx,1]**2 + targets[minIdx,2]**2 )
#                     print('min occurred at r = ', rmin)
#                     print()
#                     rmax = np.sqrt(targets[maxIdx,0]**2 + targets[maxIdx,1]**2 + targets[maxIdx,2]**2 )
#                     print('max analytic piece: ', analyticPiece[maxIdx])
#                     print('max occurred at r = ', rmax)
#                     print('~'*50)
    
                    phiNew = np.zeros((len(targets)))
#                     gpuHelmholtzConvolution[blocksPerGrid, threadsPerBlock](targets,sources,phiNew,k) 
                    gpuHelmholtzConvolutionSubractSingularity[blocksPerGrid, threadsPerBlock](targets,sources,phiNew,k) 
#                     k2 = 5
#                     gpuHelmholtzConvolutionSubractSingularity_k2[blocksPerGrid, threadsPerBlock](targets,sources,phiNew,k,k2) 
                    
                    orbitals[:,m] = np.copy(phiNew)

#                     minIdxIn = np.argmin(sources[:,3])  
#                     maxIdxIn = np.argmax(sources[:,3]) 
#                     minIdxOut = np.argmin(phiNew)  
#                     maxIdxOut = np.argmax(phiNew) 
#                     
#                     print('phi%i'%m)
#                     print('input min:  ', sources[minIdxIn,3])
#                     print('output min: ', phiNew[minIdxOut])
#                     print('input max:  ', sources[maxIdxIn,3])
#                     print('output max: ', phiNew[maxIdxOut])
#                     print('input min occurred at x,y,z = ', sources[minIdxIn,0:3])
#                     print('input max occurred at x,y,z = ', sources[maxIdxIn,0:3])
#                     print('output min occurred at x,y,z = ', sources[minIdxOut,0:3])
#                     print('output max occurred at x,y,z = ', sources[maxIdxOut,0:3])


            orthonormalizedOrbitals = modifiedGramSchrmidt(orbitals,weights)
            for m in range(tree.nOrbitals):
                normDiff = np.sqrt( np.sum( (orthonormalizedOrbitals[:,m]-oldOrbitals[:,m])**2*weights ) )
                print('Residual for orbtital %i: %1.5e' %(m,normDiff))
                if normDiff > orbitalResidual:
                    orbitalResidual = np.copy(normDiff) 
                tree.importPhiOnLeaves(orthonormalizedOrbitals[:,m], m)
#             tree.orthonormalizeOrbitals()
            print()
            print()
            tree.updateOrbitalEnergies()
#             tree.orthonormalizeOrbitals()

#             print('Before Veff update')
            newOrbitalEnergies = np.sum(tree.orbitalEnergies)
            bandEnergyResidual = newOrbitalEnergies - oldOrbitalEnergies
            oldOrbitalEnergies = np.copy(newOrbitalEnergies)
            
            tree.computeBandEnergy()
            eigensolveCount += 1
#             print('Sum of orbital energies after %i iterations in SCF #%i:  %f' %(eigensolveCount,greenIterationCounter,newOrbitalEnergies))
#             print()
            print('Band energy after %i iterations in SCF #%i:  %1.6f H, %1.2e H' 
                  %(eigensolveCount,greenIterationCounter,tree.totalBandEnergy, tree.totalBandEnergy-Eband))
            print('Band energy residual: ', bandEnergyResidual)
            print()
#             orbitalResidual = abs(bandEnergyResidual)

        tree.orthonormalizeOrbitals()
        tree.updateDensityAtQuadpoints()
        tree.normalizeDensity()
        sources = tree.extractLeavesDensity()  # extract the source point locations.  Currently, these are just all the leaf midpoints
        targets = np.copy(sources)

        """ 
        Compute new electron-electron potential and update pointwise potential values 
        """
        startCoulombConvolutionTime = timer()
        V_coulombNew = np.zeros((len(targets)))
        gpuPoissonConvolution[blocksPerGrid, threadsPerBlock](targets,sources,V_coulombNew)  # call the GPU convolution 
        ###gpuPoissonConvolutionSingularitySubtract[blocksPerGrid, threadsPerBlock](targets,sources,V_coulombNew,5)  # call the GPU convolution 
        tree.importVcoulombOnLeaves(V_coulombNew)
        tree.updateVxcAndVeffAtQuadpoints()
        CoulombConvolutionTime = timer() - startCoulombConvolutionTime
        print('Computing Vcoulomb and updating Veff took:    %.4f seconds. ' %CoulombConvolutionTime)

        """ 
        Compute the new orbital and total energies 
        """
        startEnergyTime = timer()
        tree.updateOrbitalEnergies(correctPositiveEnergies=False) 
        tree.updateTotalEnergy() 
        print('Band energies after Veff update: %1.6f H, %1.2e H'
              %(tree.totalBandEnergy, tree.totalBandEnergy-Eband))
        
        for m in range(tree.nOrbitals):
            if tree.orbitalEnergies[m] > 0:
                tree.scrambleOrbital(m)
                print('Scrambling orbital %i'%m)
        tree.updateOrbitalEnergies()
        
#         energyUpdateTime = timer() - startEnergyTime
#         print('Energy Update took:                     %.4f seconds. ' %energyUpdateTime)
        energyResidual = abs(Eold - tree.E)  # Compute the energyResidual for determining convergence
#         energyResidual = -10 # STOP AFTER FIRST SCF
        Eold = np.copy(tree.E)
        
        
        
        """
        Print results from current iteration
        """
#         print('Orbital Kinetic:   ', tree.orbitalKinetic)
#         print('Orbital Potential: ', tree.orbitalPotential)
        if tree.nOrbitals ==1:
            print('Orbital Energy:                        %.10f H' %(tree.orbitalEnergies) )
#         print('Orbital Energy:                         %.10f H, %.10e H' %(tree.orbitalEnergies[0],tree.orbitalEnergies[1]) )
        elif tree.nOrbitals==2:
            print('Orbital Energies:                      %.10f H, %.10f H' %(tree.orbitalEnergies[0],tree.orbitalEnergies[1]) )
        else: 
            print('Orbital Energies: ', tree.orbitalEnergies) 

        print('Updated V_coulomb:                      %.10f Hartree' %tree.totalVcoulomb)
        print('Updated V_x:                           %.10f Hartree' %tree.totalVx)
        print('Updated V_c:                           %.10f Hartree' %tree.totalVc)
        print('Updated E_x:                           %.10f H, %.10e H' %(tree.totalEx, tree.totalEx-ExTrue) )
        print('Updated E_c:                           %.10f H, %.10e H' %(tree.totalEc, tree.totalEc-EcTrue) )
        print('Updated Band Energy:                   %.10f H, %.10e H' %(tree.totalBandEnergy, tree.totalBandEnergy-Eband) )
        print('Total Energy:                          %.10f H, %.10e H' %(tree.E, tree.E-Etrue))
        print('Energy Residual:                        %.3e\n\n' %energyResidual)

#         if vtkExport != False:
#             tree.exportGreenIterationOrbital(vtkExport,greenIterationCounter)

        printEachIteration=True
#         iterationOutFile = 'iterationConvergenceLi_800.csv'
#         iterationOutFile = 'iterationConvergenceLi_1200_domain24.csv'
#         iterationOutFile = 'iterationConvergenceLi_smoothingBoth.csv'
#         iterationOutFile = 'iterationConvergenceBe_LW3_1200_perturbed.csv'
        if printEachIteration==True:
            header = ['Iteration', 'orbitalEnergies', 'exchangePotential', 'correlationPotential', 
                      'bandEnergy','exchangeEnergy', 'correlationEnergy', 'totalEnergy']
        
            myData = [greenIterationCounter, tree.orbitalEnergies, tree.totalVx, tree.totalVc, 
                      tree.totalBandEnergy, tree.totalEx, tree.totalEc, tree.E]
            
        
            if not os.path.isfile(iterationOutFile):
                myFile = open(iterationOutFile, 'a')
                with myFile:
                    writer = csv.writer(myFile)
                    writer.writerow(header) 
                
            
            myFile = open(iterationOutFile, 'a')
            with myFile:
                writer = csv.writer(myFile)
                writer.writerow(myData)
                
        """ END WRITING INDIVIDUAL ITERATION TO FILE """
        
        if greenIterationCounter%2==0:
            if onTheFlyRefinement==True:
                tree.refineOnTheFly(divideTolerance=0.05)
                if vtkExport != False:
                    filename = vtkExport + '/mesh%03d'%greenIterationCounter + '.vtk'
                    tree.exportMeshVTK(filename)
        
        if tree.E > 0.0:                       # Check that the current guess for energy didn't go positive.  Reset it if it did. 
            print('Warning, Energy is positive')
            tree.E = -0.5
            
        greenIterationCounter+=1

        
    print('\nConvergence to a tolerance of %f took %i iterations' %(interScfTolerance, greenIterationCounter))
      
# def greenIterations_KohnSham_SINGSUB(tree, intraScfTolerance, interScfTolerance, numberOfTargets, 
#                                 subtractSingularity, smoothingN, smoothingEps, auxiliaryFile='',iterationOutFile='iterationConvergence.csv',
#                                 onTheFlyRefinement = False, vtkExport=False, outputErrors=False): 
#     '''
#     Green Iterations for Kohn-Sham DFT using Clenshaw-Curtis quadrature.
#     '''
#     threadsPerBlock = 512
#     blocksPerGrid = (numberOfTargets + (threadsPerBlock - 1)) // threadsPerBlock  # compute the number of blocks based on N and threadsPerBlock
#     
#     print('\nEntering greenIterations_KohnSham_SINGSUB()')
#     print('\nNumber of targets:   ', numberOfTargets)
#     print('Threads per block:   ', threadsPerBlock)
#     print('Blocks per grid:     ', blocksPerGrid)
#     
#     greenIterationCounter=1                                     # initialize the counter to counter the number of iterations required for convergence
#     energyResidual = 1                                    # initialize the energyResidual to something that fails the convergence tolerance
#     Eold = -0.5 + tree.gaugeShift
# 
#     [Etrue, ExTrue, EcTrue, Eband] = np.genfromtxt(auxiliaryFile)[:4]
#      
# 
#     tree.orthonormalizeOrbitals()
#     tree.updateDensityAtQuadpoints()
#     tree.normalizeDensity()
# 
#     targets = tree.extractLeavesDensity()  
#     sources = tree.extractLeavesDensity() 
# 
#     V_coulombNew = np.zeros((len(targets)))
#     gpuPoissonConvolution[blocksPerGrid, threadsPerBlock](targets,sources,V_coulombNew)  # call the GPU convolution 
#     tree.importVcoulombOnLeaves(V_coulombNew)
#     tree.updateVxcAndVeffAtQuadpoints()
#     
#     tree.updateOrbitalEnergies()
#     
#     print('Set initial v_eff using orthonormalized orbitals...')
#     print('Initial kinetic:   ', tree.orbitalKinetic)
#     print('Initial Potential: ', tree.orbitalPotential)
#     
#     
# #     tree.orbitalEnergies[0] = Eold
#     
#     if vtkExport != False:
#         filename = vtkExport + '/mesh%03d'%(greenIterationCounter-1) + '.vtk'
#         tree.exportMeshVTK(filename)
#         
#     
# #     oldOrbitalEnergies = 10
#     while ( energyResidual > interScfTolerance ):
#         
#         print('\nSCF Count ', greenIterationCounter)
# 
#         orbitalResidual = 10
#         eigensolveCount = 0
#         max_scfCount = 3
#         while ( ( orbitalResidual > intraScfTolerance ) and ( eigensolveCount < max_scfCount) ):
#             
#             orbitalResidual = 0.0
#             
#             """ N ORBITAL HELMHOLTZ SOLVES """
#             for m in range(tree.nOrbitals):
#                 sources = tree.extractPhi(m)
#                 targets = np.copy(sources)
#                 targets_trimmed = []
#                 for iii in range(len(targets)):
#                     r = np.sqrt(targets[iii,0]**2 + targets[iii,1]**2 + targets[iii,2]**2)
#                     if r < 0.01:
#                         targets_trimmed.append(targets[iii,:])
#                 print(np.shape(targets_trimmed))
#                 phiOld = np.copy(targets[:,3])
#                 weights = np.copy(targets[:,5])
#                 phiNew = np.zeros((len(targets_trimmed)))
#                 k = np.sqrt(-2*tree.orbitalEnergies[m])
# 
#                 phiT=np.copy(targets[:,3])
#                 vT = np.copy(targets[:,4])
#                 analyticPiece = -2*phiT*vT/k**2
#                 
# #                 print('max abs of analytic piece: ', max(abs(analyticPiece)))
#                 minIdx = np.argmin(analyticPiece)  
#                 maxIdx = np.argmax(analyticPiece) 
#                 print('~'*50)
#                 print('Orbital %i, k=%1.3f' %(m,k))
#                 print('min analytic piece: ', analyticPiece[minIdx])
#                 rmin = np.sqrt(targets[minIdx,0]**2 + targets[minIdx,1]**2 + targets[minIdx,2]**2 )
#                 print('min occurred at r = ', rmin)
#                 print()
#                 rmax = np.sqrt(targets[maxIdx,0]**2 + targets[maxIdx,1]**2 + targets[maxIdx,2]**2 )
#                 print('max analytic piece: ', analyticPiece[maxIdx])
#                 print('max occurred at r = ', rmax)
#                 print('~'*50)
# 
#                 
# #                 cpuHelmholtzSingularitySubtract(targets_trimmed,sources,phiNew,k) 
#                 cpuHelmholtzSingularitySubtract_allNumerical(targets_trimmed,sources,phiNew,k) 
# #                 tree.importPhiOnLeaves(phiNew, m)
# #                 
# #                 B = np.sqrt( np.sum( phiNew**2*weights ) )
# #                 phiNew /= B
# #                 normDiff = np.sqrt( np.sum( (phiNew-phiOld)**2*weights ) )
#                 normDiff = 100
# #                 print('Residual for orbtital %i: %1.2e' %(m,normDiff))
#                 if normDiff > orbitalResidual:
#                     orbitalResidual = np.copy(normDiff)
# #                 minIdx = np.argmin(phiNew)  
# #                 maxIdx = np.argmax(phiNew) 
# #                 print('min occured at x,y,z = ', sources[minIdx,0:3])
# #                 print('max occured at x,y,z = ', sources[maxIdx,0:3])
# #                 print('min of abs(phi20): ',min(abs(phiNew)))
#             
# #             tree.orthonormalizeOrbitals()
#             tree.updateOrbitalEnergies()
#             
# #             newOrbitalEnergies = np.sum(tree.orbitalEnergies)
# #             orbitalResidual = newOrbitalEnergies - oldOrbitalEnergies
# #             oldOrbitalEnergies = np.copy(newOrbitalEnergies)
#             
#             tree.computeBandEnergy()
#             eigensolveCount += 1
# #             print('Sum of orbital energies after %i iterations in SCF #%i:  %f' %(eigensolveCount,greenIterationCounter,newOrbitalEnergies))
#             print('Band energy after %i iterations in SCF #%i:  %1.6f H, %1.2e H' 
#                   %(eigensolveCount,greenIterationCounter,tree.totalBandEnergy, tree.totalBandEnergy-Eband))
# #             print('Residual: ', orbitalResidual)
#             print()
# 
# 
#         tree.updateDensityAtQuadpoints()
#         tree.normalizeDensity()
#         sources = tree.extractLeavesDensity()  # extract the source point locations.  Currently, these are just all the leaf midpoints
# 
# 
#         """ 
#         Compute new electron-electron potential and update pointwise potential values 
#         """
#         startCoulombConvolutionTime = timer()
#         V_coulombNew = np.zeros((len(targets)))
#         gpuPoissonConvolution[blocksPerGrid, threadsPerBlock](targets,sources,V_coulombNew)  # call the GPU convolution 
#         ######gpuPoissonConvolutionSingularitySubtract[blocksPerGrid, threadsPerBlock](targets,sources,V_coulombNew,5)  # call the GPU convolution 
#         tree.importVcoulombOnLeaves(V_coulombNew)
#         tree.updateVxcAndVeffAtQuadpoints()
#         CoulombConvolutionTime = timer() - startCoulombConvolutionTime
#         print('Computing Vcoulomb and updating Veff took:    %.4f seconds. ' %CoulombConvolutionTime)
# 
#         """ 
#         Compute the new orbital and total energies 
#         """
#         startEnergyTime = timer()
#         tree.updateOrbitalEnergies() 
#         tree.updateTotalEnergy() 
#         
#         
# #         energyUpdateTime = timer() - startEnergyTime
# #         print('Energy Update took:                     %.4f seconds. ' %energyUpdateTime)
#         energyResidual = abs(Eold - tree.E)  # Compute the energyResidual for determining convergence
#         Eold = np.copy(tree.E)
#         
#         
#         
#         """
#         Print results from current iteration
#         """
# #         print('Orbital Kinetic:   ', tree.orbitalKinetic)
# #         print('Orbital Potential: ', tree.orbitalPotential)
#         if tree.nOrbitals ==1:
#             print('Orbital Energy:                        %.10f H' %(tree.orbitalEnergies) )
# #         print('Orbital Energy:                         %.10f H, %.10e H' %(tree.orbitalEnergies[0],tree.orbitalEnergies[1]) )
#         elif tree.nOrbitals==2:
#             print('Orbital Energies:                      %.10f H, %.10f H' %(tree.orbitalEnergies[0],tree.orbitalEnergies[1]) )
#         else: 
#             print('Orbital Energies: ', tree.orbitalEnergies) 
# 
#         print('Updated V_coulomb:                      %.10f Hartree' %tree.totalVcoulomb)
#         print('Updated V_x:                           %.10f Hartree' %tree.totalVx)
#         print('Updated V_c:                           %.10f Hartree' %tree.totalVc)
#         print('Updated E_x:                           %.10f H, %.10e H' %(tree.totalEx, tree.totalEx-ExTrue) )
#         print('Updated E_c:                           %.10f H, %.10e H' %(tree.totalEc, tree.totalEc-EcTrue) )
#         print('Updated Band Energy:                   %.10f H, %.10e H' %(tree.totalBandEnergy, tree.totalBandEnergy-Eband) )
# #         print('HOMO Energy                             %.10f Hartree' %tree.orbitalEnergies[0])
# #         print('Total Energy                            %.10f Hartree' %tree.E)
# #         print('\n\nHOMO Energy                             %.10f H, %.10e H' %(tree.orbitalEnergies[-1], tree.orbitalEnergies[-1]-HOMOtrue))
# #         print('\n\nHOMO Energy                            %.10f H' %(tree.orbitalEnergies[-1]))
#         print('Total Energy:                          %.10f H, %.10e H' %(tree.E, tree.E-Etrue))
#         print('Energy Residual:                        %.3e\n\n' %energyResidual)
# 
# #         if vtkExport != False:
# #             tree.exportGreenIterationOrbital(vtkExport,greenIterationCounter)
# 
#         printEachIteration=True
# #         iterationOutFile = 'iterationConvergenceLi_800.csv'
# #         iterationOutFile = 'iterationConvergenceLi_1200_domain24.csv'
# #         iterationOutFile = 'iterationConvergenceLi_smoothingBoth.csv'
# #         iterationOutFile = 'iterationConvergenceBe_LW3_1200_perturbed.csv'
#         if printEachIteration==True:
#             header = ['Iteration', 'orbitalEnergies', 'exchangePotential', 'correlationPotential', 
#                       'bandEnergy','exchangeEnergy', 'correlationEnergy', 'totalEnergy']
#         
#             myData = [greenIterationCounter, tree.orbitalEnergies, tree.totalVx, tree.totalVc, 
#                       tree.totalBandEnergy, tree.totalEx, tree.totalEc, tree.E]
#             
#         
#             if not os.path.isfile(iterationOutFile):
#                 myFile = open(iterationOutFile, 'a')
#                 with myFile:
#                     writer = csv.writer(myFile)
#                     writer.writerow(header) 
#                 
#             
#             myFile = open(iterationOutFile, 'a')
#             with myFile:
#                 writer = csv.writer(myFile)
#                 writer.writerow(myData)
#                 
#         """ END WRITING INDIVIDUAL ITERATION TO FILE """
#         
#         if greenIterationCounter%2==0:
#             if onTheFlyRefinement==True:
#                 tree.refineOnTheFly(divideTolerance=0.05)
#                 if vtkExport != False:
#                     filename = vtkExport + '/mesh%03d'%greenIterationCounter + '.vtk'
#                     tree.exportMeshVTK(filename)
#         
#         if tree.E > 0.0:                       # Check that the current guess for energy didn't go positive.  Reset it if it did. 
#             print('Warning, Energy is positive')
#             tree.E = -0.5
#             
#         greenIterationCounter+=1
# 
#         
#     print('\nConvergence to a tolerance of %f took %i iterations' %(interScfTolerance, greenIterationCounter))      
    
    
    
    """ OLD GREEN ITERATIONS FOR SCHRODINGER EQUATION         
# def greenIterations_Schrodinger_CC(tree, energyLevel, interScfTolerance, numberOfTargets, subtractSingularity, smoothingN, smoothingEps, normalizationFactor=1, threadsPerBlock=512, visualize=False, outputErrors=False):  # @DontTrace
#     '''
#     Green Iterations for the Schrodinger Equation using Clenshaw-Curtis quadrature.
#     '''
#     
#     blocksPerGrid = (numberOfTargets + (threadsPerBlock - 1)) // threadsPerBlock  # compute the number of blocks based on N and threadsPerBlock
#     
#     print('\nEntering greenIterations_Schrodinger_CC()')
#     print('\nNumber of targets:   ', numberOfTargets)
#     print('Threads per block:   ', threadsPerBlock)
#     print('Blocks per grid:     ', blocksPerGrid)
#     
#     GIcounter=1                                     # initialize the counter to counter the number of iterations required for convergence
#     energyResidual = 1                                    # initialize the energyResidual to something that fails the convergence tolerance
#     Eold = -10.0
#     Etrue = trueEnergy(energyLevel)
#     
#     while ( energyResidual > interScfTolerance ):
#         print('\nGreen Iteration Count ', GIcounter)
#         GIcounter+=1
#         
#         startExtractionTime = timer()
#         sources = tree.extractLeavesAllGridpoints()  # extract the source point locations.  Currently, these are just all the leaf midpoints
#         targets = tree.extractLeavesAllGridpoints()  # extract the target point locations.  Currently, these are all 27 gridpoints per cell (no redundancy)
#         ExtractionTime = timer() - startExtractionTime
#         
#         psiNew = np.zeros((len(targets)))
#         k = np.sqrt(-2*tree.E)                 # set the k value coming from the current guess for the energy
#         
#         startConvolutionTime = timer()
#         if subtractSingularity==0:
#             gpuConvolution[blocksPerGrid, threadsPerBlock](targets,sources,psiNew,k)  # call the GPU convolution
#         elif subtractSingularity==1:
#             gpuHelmholtzConvolutionSubractSingularity[blocksPerGrid, threadsPerBlock](targets,sources,psiNew,k)  # call the GPU convolution 
#         ConvolutionTime = timer() - startConvolutionTime
#         
#         print('Extraction took:             %.4f seconds. ' %ExtractionTime)
#         print('Convolution took:            %.4f seconds. ' %ConvolutionTime)
# 
#         tree.importPhiOnLeaves(psiNew)         # import the new wavefunction values into the tree.
#         tree.orthonormalizeOrbitals()           # Normalize the new wavefunction 
#         tree.computeWaveErrors(energyLevel,normalizationFactor)    # Compute the wavefunction errors compared to the analytic ground state 
#         print('Convolution wavefunction errors: %.10e L2,  %.10e max' %(tree.L2NormError, tree.maxPointwiseError))
# 
#         startEnergyTime = timer()
#         tree.updateTotalEnergy()  
#         energyUpdateTime = timer() - startEnergyTime
#         print('Energy Update took:              %.4f seconds. ' %energyUpdateTime)
#         energyResidual = abs(Eold - tree.E)  # Compute the energyResidual for determining convergence
#         print('Energy Residual:                 %.3e' %energyResidual)
# 
#         Eold = tree.E
#         print('Updated Potential Value:         %.10f Hartree, %.10e error' %(tree.totalPotential, -1.0 - tree.totalPotential))
#         print('Updated Kinetic Value:           %.10f Hartree, %.10e error' %(tree.totalBandEnergy, 0.5 - tree.totalBandEnergy))
#         print('Updated Energy Value:            %.10f Hartree, %.10e error' %(tree.E, tree.E-Etrue))
#         
#         if tree.E > 0.0:                       # Check that the current guess for energy didn't go positive.  Reset it if it did. 
#             print('Warning, Energy is positive')
#             tree.E = -0.5
#         
#     print('\nConvergence to a tolerance of %f took %i iterations' %(interScfTolerance, GIcounter))
#     
# 
# def greenIterations_Schrodinger_midpt(tree, energyLevel, interScfTolerance, numberOfTargets, subtractSingularity, smoothingN, smoothingEps, normalizationFactor=1, threadsPerBlock=512, visualize=False, outputErrors=False):  # @DontTrace
#     '''
#     :param interScfTolerance: exit condition for Green Iterations, energyResidual on the total energy
#     :param energyLevel: energy level trying to compute
#     '''
#     blocksPerGrid = (numberOfTargets + (threadsPerBlock - 1)) // threadsPerBlock  # compute the number of blocks based on N and threadsPerBlock
#     print('\nNumber of targets:   ', numberOfTargets)
#     print('Threads per block:   ', threadsPerBlock)
#     print('Blocks per grid:     ', blocksPerGrid)
# 
#     GIcounter=1                                     # initialize the counter to counter the number of iterations required for convergence
#     energyResidual = 1                                    # initialize the energyResidual to something that fails the convergence tolerance
#     Eold = -10.0
#     
#     Etrue = trueEnergy(energyLevel)
#     
#     while ( energyResidual > interScfTolerance ):
#         print('\nGreen Iteration Count ', GIcounter)
#         GIcounter+=1
#         
#         startExtractionTime = timer()
#         sources = tree.extractLeavesMidpointsOnly()  # extract the source point locations.  Currently, these are just all the leaf midpoints
#         targets = tree.extractLeavesAllGridpoints()  # extract the target point locations.  Currently, these are all 27 gridpoints per cell (no redundancy)
#         ExtractionTime = timer() - startExtractionTime
# 
#         psiNew = np.zeros((len(targets)))
#         startConvolutionTime = timer()
#         k = np.sqrt(-2*tree.E)                 # set the k value coming from the current guess for the energy
# 
#         skipSingular=False  # set this to TRUE to reproduce results when I was skipping singularity.
#         gpuConvolutionSmoothing[blocksPerGrid, threadsPerBlock](targets,sources,psiNew,k,smoothingN,smoothingEps,skipSingular)
#         ConvolutionTime = timer() - startConvolutionTime
#         print('Extraction took:             %.4f seconds. ' %ExtractionTime)
#         print('Convolution took:            %.4f seconds. ' %ConvolutionTime)
# 
# 
#         tree.importPhiOnLeaves(psiNew)         # import the new wavefunction values into the tree.
#         for i in range(energyLevel):
#             tree.orthogonalizeWavefunction(i)
#         tree.orthonormalizeOrbitals()           # Normalize the new wavefunction 
#         tree.computeWaveErrors(energyLevel,normalizationFactor)    # Compute the wavefunction errors compared to the analytic ground state 
#         print('Convolution wavefunction errors: %.10e L2,  %.10e max' %(tree.L2NormError, tree.maxPointwiseError))
# 
#         startEnergyTime = timer()
#         tree.updateTotalEnergy()  
#         energyUpdateTime = timer() - startEnergyTime
#         print('Energy Update took:              %.4f seconds. ' %energyUpdateTime)
#         energyResidual = abs(Eold - tree.E)  # Compute the energyResidual for determining convergence
#         print('Energy Residual:                 %.3e' %energyResidual)
# 
#         Eold = tree.E
#         print('Updated Potential Value:         %.10f Hartree, %.10e error' %(tree.totalPotential, -1.0 - tree.totalPotential))
#         print('Updated Kinetic Value:           %.10f Hartree, %.10e error' %(tree.totalBandEnergy, 0.5 - tree.totalBandEnergy))
#         print('Updated Energy Value:            %.10f Hartree, %.10e error' %(tree.E, tree.E-Etrue))
#         
#         if tree.E > 0.0:                       # Check that the current guess for energy didn't go positive.  Reset it if it did. 
#             print('Warning, Energy is positive')
#             tree.E = -1.0
#             
#     print('\nConvergence to a tolerance of %f took %i iterations' %(interScfTolerance, GIcounter))
#     
        """