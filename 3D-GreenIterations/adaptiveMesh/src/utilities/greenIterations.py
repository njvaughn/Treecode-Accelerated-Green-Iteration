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
from numpy import float32

from hydrogenAtom import trueEnergy, trueWavefunction
from convolution import *

def greenIterations_KohnSham_SCF(tree, scfTolerance, totalEnergyTolerance, numberOfTargets, 
                                subtractSingularity, smoothingN, smoothingEps, auxiliaryFile='',normalizationFactor=1, 
                                threadsPerBlock=512, onTheFlyRefinement = False, vtkExport=False, outputErrors=False):  # @DontTrace
    '''
    Green Iterations for Kohn-Sham DFT using Clenshaw-Curtis quadrature.
    '''
    
    blocksPerGrid = (numberOfTargets + (threadsPerBlock - 1)) // threadsPerBlock  # compute the number of blocks based on N and threadsPerBlock
    
    print('\nEntering greenIterations_KohnSham_SCF()')
    print('\nNumber of targets:   ', numberOfTargets)
    print('Threads per block:   ', threadsPerBlock)
    print('Blocks per grid:     ', blocksPerGrid)
    
    greenIterationCounter=1                                     # initialize the counter to counter the number of iterations required for convergence
    residual = 1                                    # initialize the residual to something that fails the convergence tolerance
    Eold = -0.5

    """ H2 molecule """
#     Etrue = -1.1394876  # from DFT-FE,  T=1e-3
#     HOMOtrue = -0.378665

    """ Beryllium Atom """
#     Etrue = -1.4446182766680081e+01
#     ExTrue = -2.2902495359115198e+00
#     EcTrue = -2.2341044592808737e-01
#     Eband = -8.1239182420318166e+00

    """ Lithium Atom """
#     Etrue = -7.3340536782581447
#     ExTrue = -1.4916149721121696
#     EcTrue = -0.15971669832262905
#     Eband = -3.8616389456972078

    [Etrue, ExTrue, EcTrue, Eband] = np.genfromtxt(auxiliaryFile)[:4]
     

    tree.orthonormalizeOrbitals()
    tree.updateDensityAtQuadpoints()
    tree.normalizeDensity()

    targets = tree.extractLeavesDensity()  
    sources = tree.extractLeavesDensity() 

    V_coulombNew = np.zeros((len(targets)))
    gpuPoissonConvolution[blocksPerGrid, threadsPerBlock](targets,sources,V_coulombNew)  # call the GPU convolution 
#     gpuPoissonConvolutionSingularitySubtract[blocksPerGrid, threadsPerBlock](targets,sources,V_coulombNew,10)  # call the GPU convolution 
    tree.importVcoulombOnLeaves(V_coulombNew)
    tree.updateVxcAndVeffAtQuadpoints()
    
#     tree.computeOrbitalKinetics()
#     tree.computeOrbitalPotentials()
    tree.updateOrbitalEnergies()
    
    print('Set initial v_eff using orthonormalized orbitals...')
    print('Initial kinetic:   ', tree.orbitalKinetic)
    print('Initial Potential: ', tree.orbitalPotential)
    
    
#     tree.orbitalEnergies[0] = Eold
    
    if vtkExport != False:
        filename = vtkExport + '/mesh%03d'%(greenIterationCounter-1) + '.vtk'
        tree.exportMeshVTK(filename)
        
    
    while ( residual > totalEnergyTolerance ):
        
#         print('\nGreen Iteration Count ', greenIterationCounter)
        print('\nSCF Count ', greenIterationCounter)
        
#         tree.normalizeOrbital()
        
        """ 
        Extract leaves and perform the Helmholtz solve 
        """
        startExtractionTime = timer()

        scfResidual = 10
        oldOrbitalEnergies = 10
        eigensolveCount = 0
        max_scfCount = 7
        while ( ( abs(scfResidual) > scfTolerance ) and ( eigensolveCount < max_scfCount) ):


            """ TWO ORBITAL HELMHOLTZ """
            sources = tree.extractPhi(0)  # extract the source point locations.  Currently, these are just all the leaf midpoints
            targets = tree.extractPhi(0)  # extract the target point locations.  Currently, these are all 27 gridpoints per cell (no redundancy)
#             print(np.shape(sources[:,3]))
#             print('max and min of phi10: ', np.max(sources[:,3]), np.min(sources[:,3]))
            phiNew = np.zeros((len(targets)))
            k = np.sqrt(-2*tree.orbitalEnergies[0]) 
            
            if k >0.0:
                print('k = ',k,', using Singularity Subtraction for phi0.')
                gpuHelmholtzConvolutionSubractSingularity[blocksPerGrid, threadsPerBlock](targets,sources,phiNew,k)  # call the GPU convolution 
            else:
#                 print('k = ',k,', skipping singularity for phi0.')
#                 gpuConvolution[blocksPerGrid, threadsPerBlock](targets,sources,phiNew,k)  # call the GPU convolution 
                print('k = ',k,', smoothing singularity for phi0.')
                gpuConvolutionSmoothing[blocksPerGrid, threadsPerBlock](targets,sources,phiNew,k, smoothingN, smoothingEps)  # call the GPU convolution 
    
            tree.importPhiOnLeaves(phiNew, 0)         # import the new wavefunction values into the tree.
    #             print('max and min of phi10: ', max(phiNew), min(phiNew))
    #             tree.orthonormalizeOrbitals()
    #             tree.updateOrbitalEnergies()
            
            if ( (tree.nOrbitals>1) ):
#             if ( (tree.nOrbitals>1) and (greenIterationCounter > -1) ):
    #         elif greenIterationCounter%2==1:
    
                # try orthonormalizing orbitals BEFORE calling the convolution on phi1
    #             tree.orthonormalizeOrbitals()
                
                sources = tree.extractPhi(1)  
                targets = tree.extractPhi(1)
                print('max and min of phi20: ', max(sources[:,3]), min(sources[:,3]))
                minIdx = np.argmin(sources[:,3])  
                maxIdx = np.argmax(sources[:,3])  
                print('min occured at x,y,z = ', sources[minIdx,0:3])
                print('max occured at x,y,z = ', sources[maxIdx,0:3])
#                 print('min of abs(phi20): ',min(abs(sources[:,3])))
                phiNew = np.zeros((len(targets)))
                k = np.sqrt(-2*tree.orbitalEnergies[1]) 
#                 k = 0.5 
#                 print('Resetting k for testing.')
                if k>0.0:
                    print('k = ',k,', using Singularity Subtraction for phi1.')
                    gpuHelmholtzConvolutionSubractSingularity[blocksPerGrid, threadsPerBlock](targets,sources,phiNew,k)  # call the GPU convolution 
#                     print('Using Kahan summation.')
#                     gpuHelmholtzConvolutionSubractSingularity_Kahan[blocksPerGrid, threadsPerBlock](targets,sources,phiNew,k)  # call the GPU convolution 
                else:
                    if greenIterationCounter > 999:
                        print('k = ',k,', would skip, but SCF count high enough.')
                        gpuHelmholtzConvolutionSubractSingularity[blocksPerGrid, threadsPerBlock](targets,sources,phiNew,k)  # call the GPU convolution 
                    else:
#                         print('k = ',k,', skipping singularity for phi1.')
#                         gpuConvolution[blocksPerGrid, threadsPerBlock](targets,sources,phiNew,k)  # call the GPU convolution 
                        print('k = ',k,', smoothing singularity for phi1.')
                        gpuConvolutionSmoothing[blocksPerGrid, threadsPerBlock](targets,sources,phiNew,k, smoothingN, smoothingEps)  # call the GPU convolution 
    
                tree.importPhiOnLeaves(phiNew, 1)         # import the new wavefunction values into the tree.
                print('max and min of phi20: ', max(phiNew), min(phiNew))
                minIdx = np.argmin(phiNew)  
                maxIdx = np.argmax(phiNew) 
                print('min occured at x,y,z = ', sources[minIdx,0:3])
                print('max occured at x,y,z = ', sources[maxIdx,0:3])
                print('min of abs(phi20): ',min(abs(phiNew)))
    #             
    #             print('Pre-Normalization Energy')
    #             tree.updateOrbitalEnergies()
    #             tree.normalizeOrbital(0) 
    #             tree.normalizeOrbital(1) 
    #             print('After normalization:')
    #             tree.updateOrbitalEnergies()
    #             print('After orthogonalization and normalization')
                tree.orthonormalizeOrbitals()
                tree.updateOrbitalEnergies() 
            else:    
    #             tree.updateOrbitalEnergies() 
    #             tree.normalizeOrbital(0) 
                tree.orthonormalizeOrbitals()
                tree.updateOrbitalEnergies()
                 
            newOrbitalEnergies = np.sum(tree.orbitalEnergies)
            scfResidual = newOrbitalEnergies - oldOrbitalEnergies
            oldOrbitalEnergies = np.copy(newOrbitalEnergies)
            
            tree.computeBandEnergy()
            eigensolveCount += 1
#             print('Sum of orbital energies after %i iterations in SCF #%i:  %f' %(eigensolveCount,greenIterationCounter,newOrbitalEnergies))
            print('Band energy after %i iterations in SCF #%i:  %1.6f H, %1.2e H' 
                  %(eigensolveCount,greenIterationCounter,tree.totalBandEnergy, tree.totalBandEnergy-Eband))
            print('Residual: ', scfResidual)
            print()

        tree.updateDensityAtQuadpoints()
        tree.normalizeDensity()
        sources = tree.extractLeavesDensity()  # extract the source point locations.  Currently, these are just all the leaf midpoints
#         importTime = timer() - startImportTime
#         print('Import and Desnity Update took:  %.4f seconds. ' %importTime)

        """ 
        Compute new electron-electron potential and update pointwise potential values 
        """
        startCoulombConvolutionTime = timer()
        V_coulombNew = np.zeros((len(targets)))
        gpuPoissonConvolution[blocksPerGrid, threadsPerBlock](targets,sources,V_coulombNew)  # call the GPU convolution 
#         gpuPoissonConvolutionSingularitySubtract[blocksPerGrid, threadsPerBlock](targets,sources,V_coulombNew,5)  # call the GPU convolution 
        tree.importVcoulombOnLeaves(V_coulombNew)
        tree.updateVxcAndVeffAtQuadpoints()
        CoulombConvolutionTime = timer() - startCoulombConvolutionTime
        print('Computing Vcoulomb and updating Veff took:    %.4f seconds. ' %CoulombConvolutionTime)

        """ 
        Compute the new orbital and total energies 
        """
        startEnergyTime = timer()
#         tree.updateOrbitalEnergies() 
        tree.updateTotalEnergy() 
        
#         tree.E = tree.orbitalKinetic + tree.orbitalPotential
        
        
        energyUpdateTime = timer() - startEnergyTime
#         print('Energy Update took:                     %.4f seconds. ' %energyUpdateTime)
        residual = abs(Eold - tree.E)  # Compute the residual for determining convergence
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

        print('Updated V_coulomb:                      %.10f Hartree' %tree.totalVcoulomb)
        print('Updated V_x:                           %.10f Hartree' %tree.totalVx)
        print('Updated V_c:                           %.10f Hartree' %tree.totalVc)
        print('Updated E_x:                           %.10f H, %.10e H' %(tree.totalEx, tree.totalEx-ExTrue) )
        print('Updated E_c:                           %.10f H, %.10e H' %(tree.totalEc, tree.totalEc-EcTrue) )
        print('Updated Band Energy:                   %.10f H, %.10e H' %(tree.totalBandEnergy, tree.totalBandEnergy-Eband) )
#         print('HOMO Energy                             %.10f Hartree' %tree.orbitalEnergies[0])
#         print('Total Energy                            %.10f Hartree' %tree.E)
#         print('\n\nHOMO Energy                             %.10f H, %.10e H' %(tree.orbitalEnergies[-1], tree.orbitalEnergies[-1]-HOMOtrue))
#         print('\n\nHOMO Energy                            %.10f H' %(tree.orbitalEnergies[-1]))
        print('Total Energy:                          %.10f H, %.10e H' %(tree.E, tree.E-Etrue))
        print('Energy Residual:                        %.3e\n\n' %residual)

#         if vtkExport != False:
#             tree.exportGreenIterationOrbital(vtkExport,greenIterationCounter)

        printEachIteration=True
        outFile = 'iterationConvergenceLi_smoothingBoth.csv'
#         outFile = 'iterationConvergenceBe_LW3_1200_perturbed.csv'
        if printEachIteration==True:
            header = ['Iteration', 'orbitalEnergies', 'exchangePotential', 'correlationPotential', 
                      'exchangeEnergy', 'correlationEnergy', 'totalEnergy']
        
            myData = [greenIterationCounter, tree.orbitalEnergies, tree.totalVx, tree.totalVc, 
                      tree.totalEx, tree.totalEc, tree.E]
            
        
            if not os.path.isfile(outFile):
                myFile = open(outFile, 'a')
                with myFile:
                    writer = csv.writer(myFile)
                    writer.writerow(header) 
                
            
            myFile = open(outFile, 'a')
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

        
    print('\nConvergence to a tolerance of %f took %i iterations' %(totalEnergyTolerance, greenIterationCounter))
      
      
    """ OLD GREEN ITERATIONS FOR SCHRODINGER EQUATION         
# def greenIterations_Schrodinger_CC(tree, energyLevel, totalEnergyTolerance, numberOfTargets, subtractSingularity, smoothingN, smoothingEps, normalizationFactor=1, threadsPerBlock=512, visualize=False, outputErrors=False):  # @DontTrace
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
#     residual = 1                                    # initialize the residual to something that fails the convergence tolerance
#     Eold = -10.0
#     Etrue = trueEnergy(energyLevel)
#     
#     while ( residual > totalEnergyTolerance ):
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
#         residual = abs(Eold - tree.E)  # Compute the residual for determining convergence
#         print('Energy Residual:                 %.3e' %residual)
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
#     print('\nConvergence to a tolerance of %f took %i iterations' %(totalEnergyTolerance, GIcounter))
#     
# 
# def greenIterations_Schrodinger_midpt(tree, energyLevel, totalEnergyTolerance, numberOfTargets, subtractSingularity, smoothingN, smoothingEps, normalizationFactor=1, threadsPerBlock=512, visualize=False, outputErrors=False):  # @DontTrace
#     '''
#     :param totalEnergyTolerance: exit condition for Green Iterations, residual on the total energy
#     :param energyLevel: energy level trying to compute
#     '''
#     blocksPerGrid = (numberOfTargets + (threadsPerBlock - 1)) // threadsPerBlock  # compute the number of blocks based on N and threadsPerBlock
#     print('\nNumber of targets:   ', numberOfTargets)
#     print('Threads per block:   ', threadsPerBlock)
#     print('Blocks per grid:     ', blocksPerGrid)
# 
#     GIcounter=1                                     # initialize the counter to counter the number of iterations required for convergence
#     residual = 1                                    # initialize the residual to something that fails the convergence tolerance
#     Eold = -10.0
#     
#     Etrue = trueEnergy(energyLevel)
#     
#     while ( residual > totalEnergyTolerance ):
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
#         residual = abs(Eold - tree.E)  # Compute the residual for determining convergence
#         print('Energy Residual:                 %.3e' %residual)
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
#     print('\nConvergence to a tolerance of %f took %i iterations' %(totalEnergyTolerance, GIcounter))
#     
        """