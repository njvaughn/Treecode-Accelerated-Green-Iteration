"""
A CUDA version of the convolution for Green Iterations.
It is written so that each thread is responsible for one of the N target gridpoints.  
Each thread interacts with all M source midpoints. 
Note: the current implementation does not use shared memory, which would provide additional
speedup.  -- 03/19/2018 NV 
"""
# from numba import cuda
# from math import sqrt,exp,factorial,pi
# import math
import numpy as np
import os
import csv
from numba import jit
import time
# from convolution import gpuPoissonConvolution,gpuHelmholtzConvolutionSubractSingularity, cpuHelmholtzSingularitySubtract,cpuHelmholtzSingularitySubtract_allNumerical
import densityMixingSchemes as densityMixing
import sys
sys.path.append('../ctypesTests')
sys.path.append('../ctypesTests/lib')


try:
    from convolution import *
except ImportError:
    print('Unable to import JIT GPU Convolutions')
try:
    import treecodeWrapperTemplate
except ImportError:
    print('Unable to import treecodeWrapperTemplate due to ImportError')
except OSError:
    print('Unable to import treecodeWrapperTemplate due to OSError')
    
    


@jit(nopython=True,parallel=True)
def modifiedGramSchrmidt(V,weights):
    n,k = np.shape(V)
    U = np.zeros_like(V)
    U[:,0] = V[:,0] / np.dot(V[:,0],V[:,0]*weights)
    for i in range(1,k):
        U[:,i] = V[:,i]
        for j in range(i):
#             print('Orthogonalizing %i against %i' %(i,j))
            U[:,i] -= (np.dot(U[:,i],U[:,j]*weights) / np.dot(U[:,j],U[:,j]*weights))*U[:,j]
        U[:,i] /= np.dot(U[:,i],U[:,i]*weights)
        
    return U

def modifiedGramSchmidt_singleOrbital(V,weights,targetOrbital):
    n,k = np.shape(V)
    U = V[:,targetOrbital]
    for j in range(targetOrbital):
#         print('Orthogonalizing %i against %i' %(targetOrbital,j))
#         U -= (np.dot(V[:,targetOrbital],V[:,j]*weights) / np.dot(V[:,j],V[:,j]*weights))*V[:,j]
        U -= np.dot(V[:,targetOrbital],V[:,j]*weights) *V[:,j]
        U /= np.sqrt( np.dot(U,U*weights) )
    
    U /= np.sqrt( np.dot(U,U*weights) )  # normalize again at end (safegaurd for the zeroth orbital, which doesn't enter the above loop)
        
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
#         U[:,i] /= np.dot(U[:,i],U[:,i]*weights)
        
    return U

def normalizeOrbitals(V,weights):
    print('Only normalizing, not orthogonalizing orbitals')
    n,k = np.shape(V)
    U = np.zeros_like(V)
#     U[:,0] = V[:,0] / np.dot(V[:,0],V[:,0]*weights)
    for i in range(0,k):
        U[:,i]  = V[:,i]
        U[:,i] /= np.sqrt( np.dot(U[:,i],U[:,i]*weights) )
        
        if abs( 1- np.dot(U[:,i],U[:,i]*weights)) > 1e-12:
            print('orbital ', i, ' not normalized? Should be 1: ', np.dot(U[:,i],U[:,i]*weights))
    
    return U

# def wavefunctionErrors(wave1, wave2, weights, x,y,z):
#     L2error = np.sqrt( np.sum(weights*(wave1-wave2)**2 ) )
#     LinfIndex = np.argmax( abs( wave1-wave2 ))
#     Linf = wave1[LinfIndex] - wave2[LinfIndex] 
#     LinfRel = (wave1[LinfIndex] - wave2[LinfIndex])/wave1[LinfIndex]
#     xinf = x[LinfIndex]
#     yinf = y[LinfIndex]
#     zinf = z[LinfIndex]
#     
#     print('~~~~~~~~Wavefunction Errors~~~~~~~~~~')
#     print("L2 Error:             ", L2error)
#     print("Linf Error:           ", Linf)
#     print("LinfRel Error:        ", LinfRel)
#     print("Located at:           ", xinf, yinf, zinf)
#     print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')



def greenIterations_KohnSham_SCF(tree, intraScfTolerance, interScfTolerance, numberOfTargets, gradientFree, GPUpresent, mixingScheme, mixingParameter,
                                subtractSingularity, smoothingN, smoothingEps, inputFile='',outputFile='',
                                onTheFlyRefinement = False, vtkExport=False, outputErrors=False, maxOrbitals=None, maxSCFIterations=None): 
    '''
    Green Iterations for Kohn-Sham DFT using Clenshaw-Curtis quadrature.
    '''
    

    if hasattr(tree, 'referenceEigenvalues'):
        referenceEigenvalues = tree.referenceEigenvalues
    else:
        print('Tree did not have attribute referenceEigenvalues')
        referenceEigenvalues = np.zeros(tree.nOrbitals)
        return
    
    
    

    greenIterationOutFile = outputFile[:-4]+'_GREEN_'+outputFile[-4:]
    SCFiterationOutFile = outputFile[:-4]+'_SCF_'+outputFile[-4:]
    

    # Initialize density history arrays
    inputDensities = np.zeros((tree.numberOfGridpoints,1))
    outputDensities = np.zeros((tree.numberOfGridpoints,1))
    
    targets = tree.extractLeavesDensity() 
    inputDensities[:,0] = np.copy(targets[:,3])
    
    

    threadsPerBlock = 512
    blocksPerGrid = (numberOfTargets + (threadsPerBlock - 1)) // threadsPerBlock  # compute the number of blocks based on N and threadsPerBlock
    
    print('\nEntering greenIterations_KohnSham_SCF()')
    print('\nNumber of targets:   ', numberOfTargets)
    print('Threads per block:   ', threadsPerBlock)
    print('Blocks per grid:     ', blocksPerGrid)
    
    densityResidual = 10                                   # initialize the densityResidual to something that fails the convergence tolerance
    Eold = -0.5 + tree.gaugeShift

#     [Etrue, ExTrue, EcTrue, Eband] = np.genfromtxt(inputFile,dtype=[(str,str,int,int,float,float,float,float,float)])[4:8]
    [Eband, Ekinetic, Eexchange, Ecorrelation, Eelectrostatic, Etotal] = np.genfromtxt(inputFile)[4:10]
    print([Eband, Ekinetic, Eexchange, Ecorrelation, Eelectrostatic, Etotal])

    ### COMPUTE THE INITIAL HAMILTONIAN ###
    targets = tree.extractLeavesDensity()  
    sources = tree.extractLeavesDensity()
#     sources = tree.extractDenstiySecondaryMesh()   # extract density on secondary mesh

    integratedDensity = np.sum( sources[:,3]*sources[:,4] )
#     densityResidual = np.sqrt( np.sum( (sources[:,3]-oldDensity[:,3])**2*weights ) )
    print('Integrated density: ', integratedDensity)

#     startCoulombConvolutionTime = timer()
    alpha = 1
    alphasq=alpha*alpha
    
    print('Using Gaussian singularity subtraction, alpha = ', alpha)
    
    print('GPUpresent set to ', GPUpresent)
    print('Type: ', type(GPUpresent))
    if GPUpresent==False:
        numTargets = len(targets)
        numSources = len(sources)
#         print('numTargets = ', numTargets)
#         print(targets[:10,:])
#         print('numSources = ', numSources)
#         print(sources[:10,:])
        sourceX = np.copy(sources[:,0])
#         print(np.shape(sourceX))
#         print('sourceX = ', sourceX[0:10])
        sourceY = np.copy(sources[:,1])
        sourceZ = np.copy(sources[:,2])
        sourceValue = np.copy(sources[:,3])
        sourceWeight = np.copy(sources[:,4])
        
        targetX = np.copy(targets[:,0])
        targetY = np.copy(targets[:,1])
        targetZ = np.copy(targets[:,2])
        targetValue = np.copy(targets[:,3])
        targetWeight = np.copy(targets[:,4])
        
#         return
        V_coulombNew = treecodeWrapperTemplate.callCompiledC_directSum_PoissonSingularitySubtract(numTargets, numSources, alphasq, 
                                                                                                  targetX, targetY, targetZ, targetValue,targetWeight, 
                                                                                                  sourceX, sourceY, sourceZ, sourceValue, sourceWeight)
        V_coulombNew += targets[:,3]* (4*np.pi)* alphasq/2
#         print(V_coulombNew[0:10])
#         print(np.shape(V_coulombNew))
    elif GPUpresent==True:
        V_coulombNew = np.zeros((len(targets)))
        gpuHartreeGaussianSingularitySubract[blocksPerGrid, threadsPerBlock](targets,sources,V_coulombNew,alphasq)
    

#     CoulombConvolutionTime = timer() - startCoulombConvolutionTime
#     print('Computing Vcoulomb took:    %.4f seconds. ' %CoulombConvolutionTime)
    tree.importVcoulombOnLeaves(V_coulombNew)
    tree.updateVxcAndVeffAtQuadpoints()
    


    print('Update orbital energies after computing the initial Veff.  Save them as the reference values for each cell')
    tree.updateOrbitalEnergies(sortByEnergy=False, saveAsReference=True)

    tree.sortOrbitalsAndEnergies()
    print('Orbital energies after initial sort: \n', tree.orbitalEnergies)
    print('Kinetic:   ', tree.orbitalKinetic)
    print('Potential: ', tree.orbitalPotential)
    tree.updateTotalEnergy(gradientFree=False)
    """

    Print results before SCF 1
    """

    print('Orbital Energies: ', tree.orbitalEnergies) 

    print('Orbital Energy Errors after initialization: ', tree.orbitalEnergies-referenceEigenvalues[:tree.nOrbitals]-tree.gaugeShift)

    print('Updated V_x:                           %.10f Hartree' %tree.totalVx)
    print('Updated V_c:                           %.10f Hartree' %tree.totalVc)
    
    print('Updated Band Energy:                   %.10f H, %.10e H' %(tree.totalBandEnergy, tree.totalBandEnergy-Eband) )
    print('Updated Kinetic Energy:                 %.10f H, %.10e H' %(tree.totalKinetic, tree.totalKinetic-Ekinetic) )
    print('Updated E_x:                           %.10f H, %.10e H' %(tree.totalEx, tree.totalEx-Eexchange) )
    print('Updated E_c:                           %.10f H, %.10e H' %(tree.totalEc, tree.totalEc-Ecorrelation) )
    print('Updated totalElectrostatic:            %.10f H, %.10e H' %(tree.totalElectrostatic, tree.totalElectrostatic-Eelectrostatic))
    print('Total Energy:                          %.10f H, %.10e H' %(tree.E, tree.E-Etotal))
    

    
    printInitialEnergies=True

    if printInitialEnergies==True:
        header = ['Iteration', 'densityResidual', 'orbitalEnergies','bandEnergy', 'kineticEnergy', 
                  'exchangeEnergy', 'correlationEnergy', 'electrostaticEnergy', 'totalEnergy']
    
        myData = [0, 1, tree.orbitalEnergies, tree.totalBandEnergy, tree.totalKinetic, 
                  tree.totalEx, tree.totalEc, tree.totalElectrostatic, tree.E]
        
    
        if not os.path.isfile(SCFiterationOutFile):
            myFile = open(SCFiterationOutFile, 'a')
            with myFile:
                writer = csv.writer(myFile)
                writer.writerow(header) 
            
        
        myFile = open(SCFiterationOutFile, 'a')
        with myFile:
            writer = csv.writer(myFile)
            writer.writerow(myData)


    for m in range(tree.nOrbitals):
        if tree.orbitalEnergies[m] > tree.gaugeShift:
            tree.orbitalEnergies[m] = tree.gaugeShift - 1.0

    
    

    

    if vtkExport != False:
        filename = vtkExport + '/mesh%03d'%(SCFcount-1) + '.vtk'
        tree.exportGridpoints(filename)
        
    

    initialWaveData = tree.extractPhi(0)
    initialPsi0 = np.copy(initialWaveData[:,3])
    x = np.copy(initialWaveData[:,0])
    y = np.copy(initialWaveData[:,1])
    z = np.copy(initialWaveData[:,2])
    


    inputIntraSCFtolerance = np.copy(intraScfTolerance)

    
    residuals = 10*np.ones_like(tree.orbitalEnergies)
    SCFcount=0
    while ( densityResidual > interScfTolerance ):
        SCFcount += 1
        print()
        print()
        print('\nSCF Count ', SCFcount)
        print('Orbital Energies: ', tree.orbitalEnergies)
        if SCFcount > 100:
            return
        
        if SCFcount>1:
            inputDensities = np.concatenate( (inputDensities, np.reshape(targets[:,3], (tree.numberOfGridpoints,1))), axis=1)
        
     
        
        orbitals = np.zeros((len(targets),tree.nOrbitals))
        oldOrbitals = np.zeros((len(targets),tree.nOrbitals))
        
        if maxOrbitals==1:
            nOrbitals = 1
        else:
            nOrbitals = tree.nOrbitals      
        for m in range(nOrbitals):
            # fill in orbitals
            targets = tree.extractPhi(m)
            oldOrbitals[:,m] = np.copy(targets[:,3])
            orbitals[:,m] = np.copy(targets[:,3])
        

        for m in range(nOrbitals): 
            
            firstGreenIteration=True
            
            # set GI anderson mixing to false.  Only gets set to true once the orbital residual is below some tolerance.
            GIandersonMixing=False
            firstInputWavefunction=True
            firstOutputWavefunction=True
            
            if ( (tree.orbitalEnergies[m] < tree.gaugeShift) or (firstGreenIteration==True) ):
                
                firstGreenIteration = False
            
                inputWavefunctions = np.zeros((tree.numberOfGridpoints,1))
                outputWavefunctions = np.zeros((tree.numberOfGridpoints,1))
    
                orbitalResidual = 1
                eigenvalueResidual = 1
                greenIterationsCount = 1
                max_GreenIterationsCount = 1500
                
    
            
                print('Working on orbital %i' %m)
                inputIntraSCFtolerance = np.copy(intraScfTolerance)
                
   
                previousResidual = 1
                while ( ( orbitalResidual > intraScfTolerance ) and ( greenIterationsCount < max_GreenIterationsCount) ):
                
                    orbitalResidual = 0.0
    
                    sources = tree.extractPhi(m)
                    targets = np.copy(sources)
                    weights = np.copy(targets[:,5])
                    
                    oldOrbitals[:,m] = np.copy(targets[:,3])
    
                    if GIandersonMixing==True:
                        if firstInputWavefunction==True:
                            inputWavefunctions[:,0] = np.copy(oldOrbitals[:,m]) # fill first column of inputWavefunctions
                            firstInputWavefunction=False
                        else:
                            inputWavefunctions = np.concatenate( ( inputWavefunctions, np.reshape(np.copy(oldOrbitals[:,m]), (tree.numberOfGridpoints,1)) ), axis=1)
                    
    
    
                    sources = tree.extractGreenIterationIntegrand(m)
                    targets = np.copy(sources)
    

    
                    if tree.orbitalEnergies[m]<0: 
                        oldEigenvalue =  tree.orbitalEnergies[m] 
                        k = np.sqrt(-2*tree.orbitalEnergies[m])
                    
                        phiNew = np.zeros((len(targets)))
                        if subtractSingularity==0: 
                            print('Using singularity skipping')
                            gpuHelmholtzConvolution[blocksPerGrid, threadsPerBlock](targets,sources,phiNew,k) 
                        elif subtractSingularity==1:
                            if tree.orbitalEnergies[m] < -0.25: 
                                
                                
                                if GPUpresent==False:
                                    print('Using Precompiled-C Helmholtz Singularity Subtract')
                                    numTargets = len(targets)
                                    numSources = len(sources)

                                    sourceX = np.copy(sources[:,0])

                                    sourceY = np.copy(sources[:,1])
                                    sourceZ = np.copy(sources[:,2])
                                    sourceValue = np.copy(sources[:,3])
                                    sourceWeight = np.copy(sources[:,4])
                                    
                                    targetX = np.copy(targets[:,0])
                                    targetY = np.copy(targets[:,1])
                                    targetZ = np.copy(targets[:,2])
                                    targetValue = np.copy(targets[:,3])
                                    targetWeight = np.copy(targets[:,4])
                                    
                                    phiNew = treecodeWrapperTemplate.callCompiledC_directSum_HelmholtzSingularitySubtract(numTargets, numSources, k, 
                                                                                                                          targetX, targetY, targetZ, targetValue, targetWeight, 
                                                                                                                          sourceX, sourceY, sourceZ, sourceValue, sourceWeight)
                                    phiNew += 4*np.pi*targets[:,3]/k**2
                                    phiNew /= (4*np.pi)
                                else:
                                    startTime = time.time()
                                    gpuHelmholtzConvolutionSubractSingularity[blocksPerGrid, threadsPerBlock](targets,sources,phiNew,k) 
                                    convolutionTime = time.time()-startTime
                                    print('Using singularity subtraction.  Convolution time: ', convolutionTime)
                            else:
                                print('Using singularity skipping because energy too close to 0')
                                gpuHelmholtzConvolution[blocksPerGrid, threadsPerBlock](targets,sources,phiNew,k)
                        else:
                            print('Invalid option for singularitySubtraction, should be 0 or 1.')
                            return
                        
                        
                        """ Method where you dont compute kinetics, from Harrison """
                        
                        # update the energy first
                        
                
                        if ( (gradientFree==True) and (SCFcount>-1) ):
                            
                            
                            tree.importPhiNewOnLeaves(phiNew)
                            tree.updateOrbitalEnergies_NoGradients(m, newOccupations=False)
                            orbitals[:,m] = np.copy(phiNew)
                            

                            orthWavefunction = modifiedGramSchmidt_singleOrbital(orbitals,weights,m)
                            orbitals[:,m] = np.copy(orthWavefunction)
                            tree.importPhiOnLeaves(orbitals[:,m], m)

     
                            if greenIterationsCount==1:
                                eigenvalueHistory = np.array(tree.orbitalEnergies[m])
                            else:
                                eigenvalueHistory = np.append(eigenvalueHistory, tree.orbitalEnergies[m])
                            print('eigenvalueHistory: \n',eigenvalueHistory)
                            
                            
                            print('Orbital energy after Harrison update: ', tree.orbitalEnergies[m])
                            
  
                        elif ( (gradientFree==False) or (SCFcount==-1) ):
 
                            # update the orbital
                            orbitals[:,m] = np.copy(phiNew)
                            tree.importPhiOnLeaves(orbitals[:,m], m)
                            tree.orthonormalizeOrbitals(targetOrbital=m)
                            
                            tree.updateOrbitalEnergies(sortByEnergy=False, targetEnergy=m)
          
                            
                        else:
                            print('Invalid option for gradientFree, which is set to: ', gradientFree)
                            print('type: ', type(gradientFree))
        
                        newEigenvalue = tree.orbitalEnergies[m]
                
                        if newEigenvalue > tree.gaugeShift:
                            if greenIterationsCount < 10:
                                tree.orbitalEnergies[m] = tree.gaugeShift-0.5
                                print('Setting energy to gauge shift - 0.5 because new value was positive.')
                        
                            else:
                                tree.orbitalEnergies[m] = tree.gaugeShift-5.5
                                if greenIterationsCount % 10 == 0:
                                    tree.scrambleOrbital(m)
                                    tree.orthonormalizeOrbitals(targetOrbital=m)
                                    print("Scrambling orbital because it's been a multiple of 10.")
    
                    else:
                        print('Orbital %i energy greater than zero.  Not performing Green Iterations for it...' %m)
                        
                    
                    tempOrbital = tree.extractPhi(m)
                    orbitals[:,m] = tempOrbital[:,3]
                    normDiff = np.sqrt( np.sum( (orbitals[:,m]-oldOrbitals[:,m])**2*weights ) )
                    eigenvalueDiff = abs(newEigenvalue - oldEigenvalue)
                    
                    

                    residuals[m] = normDiff
                    orbitalResidual = np.copy(normDiff)
                    
                    
                    if GIandersonMixing==True:
                        
                        if firstOutputWavefunction==True:
                            outputWavefunctions[:,0] = np.copy(orbitals[:,m]) # fill first column of outputWavefunctions
                            firstOutputWavefunction=False
                        else:
                            outputWavefunctions = np.concatenate( ( outputWavefunctions, np.reshape(np.copy(orbitals[:,m]), (tree.numberOfGridpoints,1)) ), axis=1)
                        
                        
                    if vtkExport != False:
                        if m>-1:
                            filename = vtkExport + '/scf_%i_orbital_%i_iteration%03d'%(SCFcount,m,greenIterationsCount) #+ '.vtk'
                            tree.exportGridpoints(filename)

                                                
                    GIsimpleMixing=False
                    
                    if GIsimpleMixing==True:
                        print('Simple mixing on the orbital.')
                        simplyMixedOrbital =  (1-mixingParameter)*orbitals[:,m]+ mixingParameter*oldOrbitals[:,m] 
                        tree.importPhiOnLeaves(simplyMixedOrbital, m)
                        
                    if GIandersonMixing==True:
                        print('Anderson mixing on the orbital.')
                        andersonOrbital, andersonWeights = densityMixing.computeNewDensity(inputWavefunctions, outputWavefunctions, mixingParameter,weights, returnWeights=True)
                        tree.importPhiOnLeaves(andersonOrbital, m)
                        
                        

                    print('Orbital %i error and eigenvalue residual:   %1.3e and %1.3e' %(m,tree.orbitalEnergies[m]-referenceEigenvalues[m]-tree.gaugeShift, eigenvalueDiff))
                    print('Orbital %i wavefunction residual: %1.3e' %(m, orbitalResidual))
                    print()
                    print()
    
      
  
                    header = ['targetOrbital', 'Iteration', 'orbitalResiduals', 'energyEigenvalues', 'eigenvalueResidual']
            
                    myData = [m, greenIterationsCount, residuals,
                              tree.orbitalEnergies-tree.gaugeShift, eigenvalueDiff]
    
                    if not os.path.isfile(greenIterationOutFile):
                        myFile = open(greenIterationOutFile, 'a')
                        with myFile:
                            writer = csv.writer(myFile)
                            writer.writerow(header) 
                        
                    
                    myFile = open(greenIterationOutFile, 'a')
                    with myFile:
                        writer = csv.writer(myFile)
                        writer.writerow(myData)
                    
                    
                    # If wavefunction residual is low then start using Anderson Mixing
                    if ((GIandersonMixing==False) and (orbitalResidual < 1e-3) ):
                        GIandersonMixing = True
                        print('Turning on Anderson Mixing for wavefunction %i' %m)
                    ### EXIT CONDITIONS ###

                    if orbitalResidual < intraScfTolerance:
                        print('Used %i iterations for orbital %i.\n\n\n' %(greenIterationsCount,m))
                        

                        
                    previousResidual = np.copy(orbitalResidual)
                    greenIterationsCount += 1
                    
                
            else:
                print('orbital %i energy is positive, not updating it anymore.' %m)
        
        
        # sort by energy and compute new occupations
        tree.sortOrbitalsAndEnergies()
        tree.computeOccupations()
        
        
        ##  DO I HAVE ENOUGH ORBITALS?  CHECK, AND ADD ONE IF NOT.
#         if tree.occupations[-1] > 1e-5:
#              
#             print('Occupation of final state is ', tree.occupations[-1])
#             tree.increaseNumberOfWavefunctionsByOne()
#             residuals = np.append(residuals, 0.0)
#             print('Increased number of wavefunctions to ', tree.nOrbitals)
            
            


        print()
        print()


        
        if maxOrbitals==1:
            print('Not updating density or anything since only computing one of the orbitals, not all.')
            return
        

        oldDensity = tree.extractLeavesDensity()
        
        
        
        tree.updateDensityAtQuadpoints()
         
        sources = tree.extractLeavesDensity()  # extract the source point locations.  Currently, these are just all the leaf midpoints
        targets = np.copy(sources)
        newDensity = np.copy(sources[:,3])
        
        if SCFcount==1: # not okay anymore because output density gets reset when tolerances get reset.
            outputDensities[:,0] = np.copy(newDensity)
        else:
            outputDensities = np.concatenate( ( outputDensities, np.reshape(np.copy(newDensity), (tree.numberOfGridpoints,1)) ), axis=1)
            
        integratedDensity = np.sum( newDensity*weights )
        densityResidual = np.sqrt( np.sum( (sources[:,3]-oldDensity[:,3])**2*weights ) )
        print('Integrated density: ', integratedDensity)
        print('Density Residual ', densityResidual)
        
        densityResidual = np.sqrt( np.sum( (outputDensities[:,SCFcount-1] - inputDensities[:,SCFcount-1])**2*weights ) )
        print('Density Residual from arrays ', densityResidual)
        print('Shape of density histories: ', np.shape(outputDensities), np.shape(inputDensities))
        
        # Now compute new mixing with anderson scheme, then import onto tree. 
      
        
        if mixingScheme == 'Simple':
            print('Using simple mixing, from the input/output arrays')
            simpleMixingDensity = mixingParameter*inputDensities[:,SCFcount-1] + (1-mixingParameter)*outputDensities[:,SCFcount-1]
            integratedDensity = np.sum( simpleMixingDensity*weights )
            print('Integrated simple mixing density: ', integratedDensity)
            tree.importDensityOnLeaves(simpleMixingDensity)
        
        elif mixingScheme == 'Anderson':
            print('Using anderson mixing.')
            andersonDensity = densityMixing.computeNewDensity(inputDensities, outputDensities, mixingParameter,weights)
            integratedDensity = np.sum( andersonDensity*weights )
            print('Integrated anderson density: ', integratedDensity)
            tree.importDensityOnLeaves(andersonDensity)
        
        elif mixingScheme == 'None':
            pass # don't touch the density
        
        
        else:
            print('Mixing must be set to either Simple, Anderson, or None')
            return
            

 
        """ 
        Compute new electron-electron potential and update pointwise potential values 
        """
#         startCoulombConvolutionTime = timer()
        sources = tree.extractLeavesDensity()  # extract the source point locations.  Currently, these are just all the leaf midpoints
        targets = np.copy(sources)
        V_coulombNew = np.zeros((len(targets)))
        gpuHartreeGaussianSingularitySubract[blocksPerGrid, threadsPerBlock](targets,sources,V_coulombNew,alphasq)
        
      
        tree.importVcoulombOnLeaves(V_coulombNew)
        tree.updateVxcAndVeffAtQuadpoints()
#         CoulombConvolutionTime = timer() - startCoulombConvolutionTime
#         print('Computing Vcoulomb and updating Veff took:    %.4f seconds. ' %CoulombConvolutionTime)

        
        """ 
        Compute the new orbital and total energies 
        """
 
        tree.updateTotalEnergy(gradientFree=gradientFree) 
        print('Band energies after Veff update: %1.6f H, %1.2e H'
              %(tree.totalBandEnergy, tree.totalBandEnergy-Eband))
        print('Orbital Energy Errors after Veff Update: ', tree.orbitalEnergies-referenceEigenvalues[:tree.nOrbitals]-tree.gaugeShift)
        
        for m in range(tree.nOrbitals):
            print('Orbital %i error: %1.3e' %(m, tree.orbitalEnergies[m]-referenceEigenvalues[m]-tree.gaugeShift))
        
        
        energyResidual = abs( tree.E - Eold )  # Compute the energyResidual for determining convergence
        Eold = np.copy(tree.E)
        
        
        
        """
        Print results from current iteration
        """

        print('Orbital Energies: ', tree.orbitalEnergies) 

        print('Updated V_x:                           %.10f Hartree' %tree.totalVx)
        print('Updated V_c:                           %.10f Hartree' %tree.totalVc)
        
        print('Updated Band Energy:                   %.10f H, %.10e H' %(tree.totalBandEnergy, tree.totalBandEnergy-Eband) )
        print('Updated Kinetic Energy:                 %.10f H, %.10e H' %(tree.totalKinetic, tree.totalKinetic-Ekinetic) )
        print('Updated E_x:                           %.10f H, %.10e H' %(tree.totalEx, tree.totalEx-Eexchange) )
        print('Updated E_c:                           %.10f H, %.10e H' %(tree.totalEc, tree.totalEc-Ecorrelation) )
        print('Updated totalElectrostatic:            %.10f H, %.10e H' %(tree.totalElectrostatic, tree.totalElectrostatic-Eelectrostatic))
        print('Total Energy:                          %.10f H, %.10e H' %(tree.E, tree.E-Etotal))
        print('Energy Residual:                        %.3e' %energyResidual)
        print('Density Residual:                       %.3e\n\n'%densityResidual)



            
        if vtkExport != False:
            filename = vtkExport + '/mesh%03d'%(SCFcount-1) + '.vtk'
            tree.exportGridpoints(filename)

        printEachIteration=True

        if printEachIteration==True:
            header = ['Iteration', 'densityResidual', 'orbitalEnergies','bandEnergy', 'kineticEnergy', 
                      'exchangeEnergy', 'correlationEnergy', 'electrostaticEnergy', 'totalEnergy']
        
            myData = [SCFcount, densityResidual, tree.orbitalEnergies, tree.totalBandEnergy, tree.totalKinetic, 
                      tree.totalEx, tree.totalEc, tree.totalElectrostatic, tree.E]
            
        
            if not os.path.isfile(SCFiterationOutFile):
                myFile = open(SCFiterationOutFile, 'a')
                with myFile:
                    writer = csv.writer(myFile)
                    writer.writerow(header) 
                
            
            myFile = open(SCFiterationOutFile, 'a')
            with myFile:
                writer = csv.writer(myFile)
                writer.writerow(myData)
                
        """ END WRITING INDIVIDUAL ITERATION TO FILE """
     
        
        if tree.E > 0.0:                       # Check that the current guess for energy didn't go positive.  Reset it if it did. 
            print('Warning, Energy is positive')
            tree.E = -0.5
            
        
        if SCFcount >= 150:
            print('Setting density residual to -1 to exit after the 150th SCF')
            densityResidual = -1
        


        
    print('\nConvergence to a tolerance of %f took %i iterations' %(interScfTolerance, SCFcount))

