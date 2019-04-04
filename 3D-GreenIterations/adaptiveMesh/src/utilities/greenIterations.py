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
from numba import cuda, jit, njit
import time
# from convolution import gpuPoissonConvolution,gpuHelmholtzConvolutionSubractSingularity, cpuHelmholtzSingularitySubtract,cpuHelmholtzSingularitySubtract_allNumerical
import densityMixingSchemes as densityMixing
from densityMixingSchemes import AitkenAcceleration, nathanAcceleration, AitkenPointwiseAcceleration
from fermiDiracDistribution import computeOccupations
import sys
# from guppy import hpy 
import resource
sys.path.append('../ctypesTests')
sys.path.append('../ctypesTests/lib')


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
    
     
# import treecodeWrappers


@jit(parallel=True)
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

@jit()
def modifiedGramSchmidt_singleOrbital(V,weights,targetOrbital, n, k):
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

xi=yi=zi=-1.1
xf=yf=zf=1.1
numpts=3000

def greenIterations_KohnSham_SCF(tree, intraScfTolerance, interScfTolerance, numberOfTargets, gradientFree, symmetricIteration, GPUpresent, 
                                 treecode, treecodeOrder, theta, maxParNode, batchSize,
                                 mixingScheme, mixingParameter, mixingHistoryCutoff,
                                subtractSingularity, gaussianAlpha, inputFile='',outputFile='',restartFile=False,
                                onTheFlyRefinement = False, vtkExport=False, outputErrors=False, maxOrbitals=None, maxSCFIterations=None): 
    '''
    Green Iterations for Kohn-Sham DFT using Clenshaw-Curtis quadrature.
    '''
    
#     return
    print('MEMORY USAGE: ', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    print()

    if hasattr(tree, 'referenceEigenvalues'):
        referenceEigenvalues = tree.referenceEigenvalues
    else:
        print('Tree did not have attribute referenceEigenvalues')
        referenceEigenvalues = np.zeros(tree.nOrbitals)
        return
    
    # Store Tree variables locally
    numberOfGridpoints = tree.numberOfGridpoints
    gaugeShift = tree.gaugeShift
    Temperature = 200  # set to 200 Kelvin
    
    

    greenIterationOutFile = outputFile[:-4]+'_GREEN_'+str(tree.numberOfGridpoints)+outputFile[-4:]
    SCFiterationOutFile =   outputFile[:-4]+'_SCF_'+str(tree.numberOfGridpoints)+outputFile[-4:]
    densityPlotsDir =       outputFile[:-4]+'_SCF_'+str(tree.numberOfGridpoints)+'_plots'
    restartFilesDir =       '/home/njvaughn/restartFiles/'+'restartFiles_'+str(tree.numberOfGridpoints)
    wavefunctionFile =      restartFilesDir+'/wavefunctions'
    densityFile =           restartFilesDir+'/density'
    inputDensityFile =      restartFilesDir+'/inputdensity'
    outputDensityFile =     restartFilesDir+'/outputdensity'
    vHartreeFile =          restartFilesDir+'/vHartree'
    auxiliaryFile =         restartFilesDir+'/auxiliary'
    
    plotSliceOfDensity=True
    if plotSliceOfDensity==True:
        try:
            os.mkdir(densityPlotsDir)
        except OSError:
            print('Unable to make directory ', densityPlotsDir)
        
    try:
        os.mkdir(restartFilesDir)
    except OSError:
        print('Unable to make restart directory ', restartFilesDir)
    
    
    
    if maxOrbitals==1:
        nOrbitals = 1
    else:
        nOrbitals = tree.nOrbitals
            
    if restartFile!=False:
        orbitals = np.load(wavefunctionFile+'.npy')
        oldOrbitals = np.copy(orbitals)
        for m in range(nOrbitals): 
            tree.importPhiOnLeaves(orbitals[:,m], m)
        density = np.load(densityFile+'.npy')
        tree.importDensityOnLeaves(density)
        
        inputDensities = np.load(inputDensityFile+'.npy')
        outputDensities = np.load(outputDensityFile+'.npy')
        
        V_hartreeNew = np.load(vHartreeFile+'.npy')
        tree.importVhartreeOnLeaves(V_hartreeNew)
        tree.updateVxcAndVeffAtQuadpoints()
        
        
        # make and save dictionary
        auxiliaryRestartData = np.load(auxiliaryFile+'.npy').item()
        print('type of aux: ', type(auxiliaryRestartData))
        SCFcount = auxiliaryRestartData['SCFcount']
        tree.totalIterationCount = auxiliaryRestartData['totalIterationCount']
        tree.orbitalEnergies = auxiliaryRestartData['eigenvalues'] 
        Eold = auxiliaryRestartData['Eold']
    
    else: 
        Eold = -10
        SCFcount=0
        tree.totalIterationCount = 0
        
        # Initialize orbital matrix
        targets = tree.extractLeavesDensity()
        orbitals = np.zeros((len(targets),tree.nOrbitals))
        oldOrbitals = np.zeros((len(targets),tree.nOrbitals))
        
              
        for m in range(nOrbitals):
            # fill in orbitals
            targets = tree.extractPhi(m)
            weights = np.copy(targets[:,5])
            oldOrbitals[:,m] = np.copy(targets[:,3])
            orbitals[:,m] = np.copy(targets[:,3])
            
        # Initialize density history arrays
        inputDensities = np.zeros((numberOfGridpoints,1))
        outputDensities = np.zeros((numberOfGridpoints,1))
        
        targets = tree.extractLeavesDensity() 
        weights = targets[:,4]
        inputDensities[:,0] = np.copy(targets[:,3])

    targets = tree.extractLeavesDensity() 
    weights = targets[:,4]
    
        
    
        
    if plotSliceOfDensity==True:
        densitySliceSavefile = densityPlotsDir+'/densities'
        print()
        r, rho = tree.interpolateDensity(xi,yi,zi,xf,yf,zf, numpts, plot=False, save=False)
        
        densities = np.concatenate( (np.reshape(r, (numpts,1)), np.reshape(rho, (numpts,1))), axis=1)
        np.save(densitySliceSavefile,densities)

    
    

    threadsPerBlock = 512
    blocksPerGrid = (numberOfTargets + (threadsPerBlock - 1)) // threadsPerBlock  # compute the number of blocks based on N and threadsPerBlock
    
    print('\nEntering greenIterations_KohnSham_SCF()')
    print('\nNumber of targets:   ', numberOfTargets)
    print('Threads per block:   ', threadsPerBlock)
    print('Blocks per grid:     ', blocksPerGrid)
    
    densityResidual = 10                                   # initialize the densityResidual to something that fails the convergence tolerance

#     [Etrue, ExTrue, EcTrue, Eband] = np.genfromtxt(inputFile,dtype=[(str,str,int,int,float,float,float,float,float)])[4:8]
    [Eband, Ekinetic, Eexchange, Ecorrelation, Ehartree, Etotal] = np.genfromtxt(inputFile)[3:9]
    print([Eband, Ekinetic, Eexchange, Ecorrelation, Ehartree, Etotal])

    ### COMPUTE THE INITIAL HAMILTONIAN ###
    density_targets = tree.extractLeavesDensity()  
    density_sources = np.copy(density_targets)
#     sources = tree.extractDenstiySecondaryMesh()   # extract density on secondary mesh

    integratedDensity = np.sum( density_sources[:,3]*density_sources[:,4] )
#     densityResidual = np.sqrt( np.sum( (sources[:,3]-oldDensity[:,3])**2*weights ) )
    print('Integrated density: ', integratedDensity)

#     starthartreeConvolutionTime = timer()
    alpha = gaussianAlpha
    alphasq=alpha*alpha
    
    
    if restartFile==False: # need to do initial Vhartree solve
        print('Using Gaussian singularity subtraction, alpha = ', alpha)
        
        print('GPUpresent set to ', GPUpresent)
        print('Type: ', type(GPUpresent))
        if GPUpresent==False:
            numTargets = len(density_targets)
            numSources = len(density_sources)
    #         print('numTargets = ', numTargets)
    #         print(targets[:10,:])
    #         print('numSources = ', numSources)
    #         print(sources[:10,:])
            copystart = time.time()
            sourceX = np.copy(density_sources[:,0])
    #         print(np.shape(sourceX))
    #         print('sourceX = ', sourceX[0:10])
            sourceY = np.copy(density_sources[:,1])
            sourceZ = np.copy(density_sources[:,2])
            sourceValue = np.copy(density_sources[:,3])
            sourceWeight = np.copy(density_sources[:,4])
            
            targetX = np.copy(density_targets[:,0])
            targetY = np.copy(density_targets[:,1])
            targetZ = np.copy(density_targets[:,2])
            targetValue = np.copy(density_targets[:,3])
            targetWeight = np.copy(density_targets[:,4])
            copytime=time.time()-copystart
            print('Copy time before convolution: ', copytime)
            start = time.time()
            
            if treecode==False:
                V_hartreeNew = directSumWrappers.callCompiledC_directSum_PoissonSingularitySubtract(numTargets, numSources, alphasq, 
                                                                                                      targetX, targetY, targetZ, targetValue,targetWeight, 
                                                                                                      sourceX, sourceY, sourceZ, sourceValue, sourceWeight)
    
                V_hartreeNew += targets[:,3]* (4*np.pi)/ alphasq/ 2   # Correct for exp(-r*r/alphasq)  # DONT TRUST
    
            elif treecode==True:
                
                
    # #         V_hartreeNew += targets[:,3]* (4*np.pi)* alphasq/2  # Wrong
    
    
    #         V_hartreeNew = directSumWrappers.callCompiledC_directSum_Poisson(numTargets, numSources, 
    #                                                                         targetX, targetY, targetZ, targetValue,targetWeight, 
    #                                                                         sourceX, sourceY, sourceZ, sourceValue, sourceWeight)
    
                potentialType=2 # shoud be 2 for Hartree w/ singularity subtraction.  Set to 0, 1, or 3 just to test other kernels quickly
                alpha = gaussianAlpha
                V_hartreeNew = treecodeWrappers.callTreedriver(numTargets, numSources, 
                                                               targetX, targetY, targetZ, targetValue, 
                                                               sourceX, sourceY, sourceZ, sourceValue, sourceWeight,
                                                               potentialType, alpha, treecodeOrder, theta, maxParNode, batchSize)
                   
                if potentialType==2:
                    V_hartreeNew += targets[:,3]* (4*np.pi) / alphasq/2
    
            
    #         print('First few terms of V_hartreeNew: ', V_hartreeNew[:8])
            print('Convolution time: ', time.time()-start)
            
            
            
            
        elif GPUpresent==True:
            if treecode==False:
                V_hartreeNew = np.zeros((len(density_targets)))
                start = time.time()
                gpuHartreeGaussianSingularitySubract[blocksPerGrid, threadsPerBlock](density_targets,density_sources,V_hartreeNew,alphasq)
                print('Convolution time: ', time.time()-start)
    #             return
            elif treecode==True:
                copystart=time.time()
                numTargets = len(density_targets)
                numSources = len(density_sources)
                sourceX = np.copy(density_sources[:,0])
    
                sourceY = np.copy(density_sources[:,1])
                sourceZ = np.copy(density_sources[:,2])
                sourceValue = np.copy(density_sources[:,3])
                sourceWeight = np.copy(density_sources[:,4])
                
                targetX = np.copy(density_targets[:,0])
                targetY = np.copy(density_targets[:,1])
                targetZ = np.copy(density_targets[:,2])
                targetValue = np.copy(density_targets[:,3])
                targetWeight = np.copy(density_targets[:,4])
                copytime = time.time()-copystart
                print('Copy time before calling treecode: ', copytime)
                start = time.time()
                potentialType=2 
                alpha = gaussianAlpha
                V_hartreeNew = treecodeWrappers.callTreedriver(numTargets, numSources, 
                                                               targetX, targetY, targetZ, targetValue, 
                                                               sourceX, sourceY, sourceZ, sourceValue, sourceWeight,
                                                               potentialType, alpha, treecodeOrder, theta, maxParNode, batchSize)
                print('Convolution time: ', time.time()-start)
                
            else:
                print('treecode True or False?')
                return
        
    
    #     hartreeConvolutionTime = timer() - starthartreeConvolutionTime
    #     print('Computing Vhartree took:    %.4f seconds. ' %hartreeConvolutionTime)
        tree.importVhartreeOnLeaves(V_hartreeNew)
        tree.updateVxcAndVeffAtQuadpoints()
        
        
        ### Write output files that will be used to test the Treecode evaluation ###
    #     sourcesTXT = '/Users/nathanvaughn/Documents/testData/H2Sources.txt'
    #     targetsTXT = '/Users/nathanvaughn/Documents/testData/H2Targets.txt'
    #     hartreePotentialTXT = '/Users/nathanvaughn/Documents/testData/H2HartreePotential.txt'
        
    #     np.savetxt(sourcesTXT, sources)
    #     np.savetxt(targetsTXT, targets[:,0:4])
    #     np.savetxt(hartreePotentialTXT, V_hartreeNew)
    #     
    #     return
    
    
        print('Update orbital energies after computing the initial Veff.  Save them as the reference values for each cell')
        tree.updateOrbitalEnergies(sortByEnergy=False, saveAsReference=True)
        tree.computeBandEnergy()
        
        tree.sortOrbitalsAndEnergies()
        for m in range(nOrbitals):
            # fill in orbitals
            targets = tree.extractPhi(m)
            weights = np.copy(targets[:,5])
            oldOrbitals[:,m] = np.copy(targets[:,3])
            orbitals[:,m] = np.copy(targets[:,3])
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
    #     print('Updated Kinetic Energy:                 %.10f H, %.10e H' %(tree.totalKinetic, tree.totalKinetic-Ekinetic) )
        print('Updated E_H:                            %.10f H, %.10e H' %(tree.totalEhartree, tree.totalEhartree-Ehartree) )
        print('Updated E_x:                           %.10f H, %.10e H' %(tree.totalEx, tree.totalEx-Eexchange) )
        print('Updated E_c:                           %.10f H, %.10e H' %(tree.totalEc, tree.totalEc-Ecorrelation) )
    #     print('Updated totalElectrostatic:            %.10f H, %.10e H' %(tree.totalElectrostatic, tree.totalElectrostatic-Eelectrostatic))
        print('Total Energy:                          %.10f H, %.10e H' %(tree.E, tree.E-Etotal))
        
        
        
        printInitialEnergies=True
    
        if printInitialEnergies==True:
            header = ['Iteration', 'densityResidual', 'orbitalEnergies','bandEnergy', 'kineticEnergy', 
                      'exchangeEnergy', 'correlationEnergy', 'hartreeEnergy', 'totalEnergy']
        
            myData = [0, 1, tree.orbitalEnergies, tree.totalBandEnergy, tree.totalKinetic, 
                      tree.totalEx, tree.totalEc, tree.totalEhartree, tree.E]
            
        
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
            
        
    #     if GPUpresent==False:
    #         print('Exiting after initialization because no GPU present.')
    #         return

    initialWaveData = tree.extractPhi(0)
    initialPsi0 = np.copy(initialWaveData[:,3])
    x = np.copy(initialWaveData[:,0])
    y = np.copy(initialWaveData[:,1])
    z = np.copy(initialWaveData[:,2])
    


    inputIntraSCFtolerance = np.copy(intraScfTolerance)

    energyResidual=1
    
    residuals = 10*np.ones_like(tree.orbitalEnergies)
    
    while ( (densityResidual > interScfTolerance) or (energyResidual > interScfTolerance) ):  # terminate SCF when both energy and density are converged.
        SCFcount += 1
        print()
        print()
        print('\nSCF Count ', SCFcount)
        print('Orbital Energies: ', tree.orbitalEnergies)
#         if SCFcount > 0:
#             print('Exiting before first SCF (for testing initialized mesh accuracy)')
#             return
        
        if SCFcount>1:
            targets = tree.extractLeavesDensity()
            
#             if SCFcount < mixingHistoryCutoff:
#             inputDensities = np.concatenate( (inputDensities, np.reshape(targets[:,3], (numberOfGridpoints,1))), axis=1)
#             else:
#                 inputDensities

            if (SCFcount-1)<mixingHistoryCutoff:
                inputDensities = np.concatenate( (inputDensities, np.reshape(targets[:,3], (numberOfGridpoints,1))), axis=1)
                print('Concatenated inputDensity.  Now has shape: ', np.shape(inputDensities))
            else:
                print('Beyond mixingHistoryCutoff.  Replacing column ', (SCFcount-1)%mixingHistoryCutoff)
#                                 print('Shape of oldOrbitals[:,m]: ', np.shape(oldOrbitals[:,m]))
                inputDensities[:,(SCFcount-1)%mixingHistoryCutoff] = np.copy(targets[:,3])
        
     
        
    
            
        

        for m in range(nOrbitals): 
            if m==2:
                print('Saving restart files for after the psi0 and psi1 complete.')
                # save arrays 
                try:
                    np.save(wavefunctionFile, orbitals)
                    
                    sources = tree.extractLeavesDensity()
                    np.save(densityFile, sources[:,3])
                    np.save(outputDensityFile, outputDensities)
                    np.save(inputDensityFile, inputDensities)
                    
                    np.save(vHartreeFile, V_hartreeNew)
                    
                    
                    
                    # make and save dictionary
                    auxiliaryRestartData = {}
                    auxiliaryRestartData['SCFcount'] = SCFcount
                    auxiliaryRestartData['totalIterationCount'] = tree.totalIterationCount
                    auxiliaryRestartData['eigenvalues'] = tree.orbitalEnergies
                    auxiliaryRestartData['Eold'] = Eold
            
                    np.save(auxiliaryFile, auxiliaryRestartData)
                except FileNotFoundError:
                    print('Failed to save restart files.')
            
            if m==3:
                print('Scrambling orbital 3...')
                tree.scrambleOrbital(m)
                tree.orthonormalizeOrbitals(targetOrbital=m)
            # Orthonormalize orbital m before beginning Green's iteration
            targets = tree.extractPhi(m)
            orbitals[:,m] = np.copy(targets[:,3])
            n,k = np.shape(orbitals)
            orthWavefunction = modifiedGramSchmidt_singleOrbital(orbitals,weights,m, n, k)
            orbitals[:,m] = np.copy(orthWavefunction)
            tree.importPhiOnLeaves(orbitals[:,m], m)
            
            inputEigenvalues = []
            
            if symmetricIteration==False:
                tree.setPhiOldOnLeaves(m)
            elif symmetricIteration==True:
                sqrtV = tree.extractSqrtV()
                psi = targets[:,3]
                
                psiSqrtV = psi*sqrtV
                psiSqrtV /= np.sqrt( np.sum(psiSqrtV*psiSqrtV*weights ))
                tree.setPhiOldOnLeaves_symmetric(psiSqrtV)
            
            firstGreenIteration=True
            
            # set GI anderson mixing to false.  Only gets set to true once the orbital residual is below some tolerance.
            GIandersonMixing=False
            firstInputWavefunction=True
            firstOutputWavefunction=True
            
            aitkenAcceleration=False
            freezeEigenvalue=False
            
            if ( (tree.orbitalEnergies[m] < tree.gaugeShift) or (firstGreenIteration==True) ):
                
                firstGreenIteration = False
            
                inputWavefunctions = np.zeros((numberOfGridpoints+1,1))
                outputWavefunctions = np.zeros((numberOfGridpoints+1,1))
                
#                 inputEigenvalues  = np.zeros((1,))
#                 outputEigenvalues = np.zeros((1,))
    
                orbitalResidual = 1
                eigenvalueResidual = 1
                greenIterationsCount = 1
                max_GreenIterationsCount = 15000  # set very high.  Don't ever stop green iterations before convergence.
                
    
            
                print('Working on orbital %i' %m)
                inputIntraSCFtolerance = np.copy(intraScfTolerance)
                orbitalResidual = 1.0
                oldOrbitalResidual = 2.0
                oldOldOrbitalResidual = 4.0
                psiNewNorm = 10
                previousResidual = 1
                previousEigenvalueDiff = 1
                
                normDiff=1.0
                eigenvalueDiff=1.0
                
                psiA = None
                psiB = None
                psiC = None
                aitkenPsi = None
                oldAitkenPsi = None
                aitkenEig = None
                oldAitkenEig = None
                
                ratioTol = 5e-30
                
                previousResidualRatio = 2
                previousEigenvalueResidualRatio = 2
                    
                while ( ( orbitalResidual > intraScfTolerance ) and ( greenIterationsCount < max_GreenIterationsCount) 
#                         and (np.abs(psiNewNorm-1) > intraScfTolerance) 
                        and (np.abs(oldOrbitalResidual-orbitalResidual)/np.abs(oldOrbitalResidual) > -1/10000)
                        and (abs(eigenvalueDiff) > -intraScfTolerance/10000) ):
                    print('MEMORY USAGE: ', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss )
                    tree.totalIterationCount += 1
#                     orbitalResidual = 0.0
                    
                    oldOldOrbitalResidual = oldOrbitalResidual
                    oldOrbitalResidual = orbitalResidual
                    
      
                    sources = tree.extractPhi(m)
                    targets = np.copy(sources)
                    weights = np.copy(targets[:,5])
                    
                    
                            
                            
                    oldOrbitals[:,m] = np.copy(targets[:,3])
    
                    if GIandersonMixing==True:
                        if firstInputWavefunction==True:
                            temp = np.append( oldOrbitals[:,m], tree.orbitalEnergies[m])
                            inputWavefunctions[:,0] = np.copy(temp) # fill first column of inputWavefunctions
#                             inputEigenvalues[0] = tree.orbitalEnergies[m]
                            firstInputWavefunction=False
                        else:
                            if (greenIterationsCount-1-mixingStart)<mixingHistoryCutoff:
                                temp = np.append( oldOrbitals[:,m], tree.orbitalEnergies[m])
                                inputWavefunctions = np.concatenate( ( inputWavefunctions, np.reshape(np.copy(temp), (numberOfGridpoints+1,1)) ), axis=1)
                                print('Concatenated inputWavefunction.  Now has shape: ', np.shape(inputWavefunctions))
#                                 inputEigenvalues = np.concatenate( inputEigenvalues, tree.orbitalEnergies[m])
#                                 print('Concatenated inputeEigenvalues.  Now has shape: ', np.shape(inputeEigenvalues))
                            else:
                                print('Beyond mixingHistoryCutoff.  Replacing column ', (greenIterationsCount-1-mixingStart)%mixingHistoryCutoff)
#                                 print('Shape of oldOrbitals[:,m]: ', np.shape(oldOrbitals[:,m]))
                                temp = np.append( oldOrbitals[:,m], tree.orbitalEnergies[m])
                                inputWavefunctions[:,(greenIterationsCount-1-mixingStart)%mixingHistoryCutoff] = np.copy(temp)
#                                 inputEigenvalues[(greenIterationsCount-1-mixingStart)%mixingHistoryCutoff] = np.copy(tree.orbitalEnergies[m])
    
    
                    if symmetricIteration==False:
#                         sources = tree.extractGreenIterationIntegrand(m)
                        sources = tree.extractNathanIterationIntegrand(m)
                    elif symmetricIteration == True:
#                     sources = tree.extractGreenIterationIntegrand_Deflated(m,orbitals,weights)
                        sources, sqrtV = tree.extractGreenIterationIntegrand_symmetric(m)
                    else: 
                        print("symmetricIteration variable not True or False.  What should it be?")
                        return
                    
                    targets = np.copy(sources)
    

    
                    if tree.orbitalEnergies[m]<0: 
                        oldEigenvalue =  tree.orbitalEnergies[m] 
                        k = np.sqrt(-2*tree.orbitalEnergies[m] - 1 )
                        print('Shifting k by one... attempt at shifted inverse iteration.')
                    
                        phiNew = np.zeros((len(targets)))
                        if subtractSingularity==0: 
                            print('Using singularity skipping')
                            gpuHelmholtzConvolution[blocksPerGrid, threadsPerBlock](targets,sources,phiNew,k) 
                        elif subtractSingularity==1:
                            if tree.orbitalEnergies[m] < 10.25: 
                                
                                
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
                                    
                                    phiNew = directSumWrappers.callCompiledC_directSum_HelmholtzSingularitySubtract(numTargets, numSources, k, 
                                                                                                                          targetX, targetY, targetZ, targetValue, targetWeight, 
                                                                                                                          sourceX, sourceY, sourceZ, sourceValue, sourceWeight)
                                    phiNew += 4*np.pi*targets[:,3]/k**2

#                                     phiNew = directSumWrappers.callCompiledC_directSum_Helmholtz(numTargets, numSources, k, 
#                                                                                               targetX, targetY, targetZ, targetValue, targetWeight, 
#                                                                                               sourceX, sourceY, sourceZ, sourceValue, sourceWeight)


#                                     potentialType=3
#                                     order=3
#                                     kappa = k
#                                     theta = 0.5
#                                     maxParNode = 500
#                                     batchSize = 500
#                                     start = time.time()
#                                     phiNew = treecodeWrappers.callTreedriver(numTargets, numSources, 
#                                                                                    targetX, targetY, targetZ, targetValue, 
#                                                                                    sourceX, sourceY, sourceZ, sourceValue, sourceWeight,
#                                                                                    potentialType, kappa, order, theta, maxParNode, batchSize)
#                                     print('Convolution time: ', time.time()-start)
#                                     if potentialType==3:
#                                         phiNew += 4*np.pi*targets[:,3]/k**2
                                    phiNew /= (4*np.pi)
                                elif GPUpresent==True:
                                    if treecode==False:
                                        startTime = time.time()
                                        if symmetricIteration==False:
                                            gpuHartreeGaussianSingularitySubract[blocksPerGrid, threadsPerBlock](targets,sources,phiNew,alphasq) 
#                                             gpuHelmholtzConvolutionSubractSingularity[blocksPerGrid, threadsPerBlock](targets,sources,phiNew,k) 

                                            convolutionTime = time.time()-startTime
                                            print('Using asymmetric singularity subtraction.  Convolution time: ', convolutionTime)
                                        elif symmetricIteration==True:
                                            gpuHelmholtzConvolutionSubractSingularitySymmetric[blocksPerGrid, threadsPerBlock](targets,sources,sqrtV,phiNew,k) 
                                            phiNew *= -1
                                            convolutionTime = time.time()-startTime
                                            print('Using symmetric singularity subtraction.  Convolution time: ', convolutionTime)

                                        
                                    elif treecode==True:
                                        
                                        copyStart = time.time()
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
                                    
                                        copytime=time.time()-copyStart
#                                         print('Time spent copying arrays for treecode call: ', copytime)
                                        potentialType=3
                                        kappa = k
                                        start = time.time()
                                        phiNew = treecodeWrappers.callTreedriver(numTargets, numSources, 
                                                                                       targetX, targetY, targetZ, targetValue, 
                                                                                       sourceX, sourceY, sourceZ, sourceValue, sourceWeight,
                                                                                       potentialType, kappa, treecodeOrder, theta, maxParNode, batchSize)
                                        print('Convolution time: ', time.time()-start)
                                        phiNew /= (4*np.pi)
                                    
                                    else: 
                                        print('treecode true or false?')
                                        return
                            else:
                                print('Using singularity skipping because energy too close to 0')
                                gpuHelmholtzConvolution[blocksPerGrid, threadsPerBlock](targets,sources,phiNew,k)
                        else:
                            print('Invalid option for singularitySubtraction, should be 0 or 1.')
                            return
                        
                        
                        """ Method where you dont compute kinetics, from Harrison """
                        
                        # update the energy first
                        
                
                        if ( (gradientFree==True) and (SCFcount>-1) and (freezeEigenvalue==False) ):
                            
                            psiNewNorm = np.sqrt( np.sum( phiNew*phiNew*weights))
                            
                            if symmetricIteration==False:
                                tree.importPhiNewOnLeaves(phiNew)
#                                 print('Not updating energy, just for testing Steffenson method')
                                tree.updateOrbitalEnergies_NoGradients(m, newOccupations=False)
                                orbitals[:,m] = np.copy(phiNew)
                            elif symmetricIteration==True:
#                                 tree.importPhiNewOnLeaves(phiNew/sqrtV)
#                                 tree.updateOrbitalEnergies_NoGradients(m, newOccupations=False)


                                # import phiNew and compute eigenvalue update
                                tree.importPhiNewOnLeaves(phiNew)
                                tree.updateOrbitalEnergies_NoGradients(m, newOccupations=False, symmetric=True)
                                
                                
#                                 print('Shifting eig up 1 because of inverse iteration attempt')
                                # Import normalized psi*sqrtV into phiOld
                                phiNew /= np.sqrt( np.sum(phiNew*phiNew*weights ))
                                tree.setPhiOldOnLeaves_symmetric(phiNew)
                                
                                
                                orbitals[:,m] = np.copy(phiNew/sqrtV)
                            
                            n,k = np.shape(orbitals)
                            orthWavefunction = modifiedGramSchmidt_singleOrbital(orbitals,weights,m, n, k)
                            orbitals[:,m] = np.copy(orthWavefunction)
                            
                            print('Single value of orbital: ', orbitals[100000,m])
                            
#                             simplyMixedWavefunction = 1/2* ( orbitals[:,m]+ oldOrbitals[:,m])
                            tree.importPhiOnLeaves(orbitals[:,m], m)
#                             tree.importPhiOnLeaves(simplyMixedWavefunction, m)
                            if symmetricIteration==False:
                                tree.setPhiOldOnLeaves(m)
                            else: # already imported the normalized psi*sqrtV
                                pass

     
                            if greenIterationsCount==1:
                                eigenvalueHistory = np.array(tree.orbitalEnergies[m])
                            else:
                                eigenvalueHistory = np.append(eigenvalueHistory, tree.orbitalEnergies[m])
                            print('eigenvalueHistory: \n',eigenvalueHistory)
                            
                            
                            print('Orbital energy after Harrison update: ', tree.orbitalEnergies[m])
                            
  
                        elif ( (gradientFree==False) or (SCFcount==-1) ):
 
                            # update the orbital
                            if symmetricIteration==False:
                                orbitals[:,m] = np.copy(phiNew)
                            if symmetricIteration==True:
                                orbitals[:,m] = np.copy(phiNew/sqrtV)
                                
                            n,k = np.shape(orbitals)
                            orthWavefunction = modifiedGramSchmidt_singleOrbital(orbitals,weights,m, n, k)
                            orbitals[:,m] = np.copy(orthWavefunction)
                            tree.importPhiOnLeaves(orbitals[:,m], m)
                            print('Single value of orbital: ', orbitals[100000,m])
#                             tree.importPhiOnLeaves(orbitals[:,m], m)
#                             tree.orthonormalizeOrbitals(targetOrbital=m)
                            
                            tree.updateOrbitalEnergies(sortByEnergy=False, targetEnergy=m)
          
                            
                        else:
                            print('Not updating eigenvalue.  Is that intended?')
#                             print('Invalid option for gradientFree, which is set to: ', gradientFree)
#                             print('type: ', type(gradientFree))
                            
                            orbitals[:,m] = np.copy(phiNew)
                            n,k = np.shape(orbitals)
                            orthWavefunction = modifiedGramSchmidt_singleOrbital(orbitals,weights,m, n, k)
                            orbitals[:,m] = np.copy(orthWavefunction)
                            tree.importPhiOnLeaves(orbitals[:,m], m)
                            tree.setPhiOldOnLeaves(m)
                            
                            print('Setting freezeEigenvalue to false')
                            freezeEigenvalue=False
                            eigenvalueHistory = np.append(eigenvalueHistory, tree.orbitalEnergies[m])
                            
        
                        newEigenvalue = tree.orbitalEnergies[m]
                        
                        
                        
                
#                         if newEigenvalue > tree.gaugeShift:
                        if newEigenvalue > 0.0:
                            if greenIterationsCount < 10:
                                tree.orbitalEnergies[m] = tree.gaugeShift-0.5
                                GIandersonMixing=False
                                print('Setting energy to gauge shift - 0.5 because new value was positive.')
                        
                            else:
                                tree.orbitalEnergies[m] = tree.gaugeShift
                                if greenIterationsCount % 10 == 0:
                                    tree.scrambleOrbital(m)
                                    tree.orthonormalizeOrbitals(targetOrbital=m)
                                    GIandersonMixing=False
                                    print("Scrambling orbital because it's been a multiple of 10.")
    
                    else:
                        print('Orbital %i energy greater than zero.  Not performing Green Iterations for it...' %m)
                        
                    
                    ##########################################################################################
                    ## Aitken Acceleration Section
                    ##########################################################################################
                    
                    tempOrbital = tree.extractPhi(m)
#                     aitkenStart=2*0  # keep this a multiple of 2 for now
                    if aitkenAcceleration==True: 
                        
                        orbitals[:,m] = np.copy( tempOrbital[:,3] )
                        normDiff = np.sqrt( np.sum( (orbitals[:,m]-oldOrbitals[:,m])**2*weights ) )
                        eigenvalueDiff = abs(newEigenvalue - oldEigenvalue)
                        
                        if (greenIterationsCount-aitkenStart)%4==1:
                            print('Saving psiA from oldOrbitals array')                            
                            psiA = np.copy(oldOrbitals[:,m])
                            print('Saving psiB from newly computed phi')
                            psiB = np.copy(tempOrbital[:,3])
#                             print('sum psiA : ', np.sum(psiA))
                            eigA = np.copy(oldEigenvalue)
                            eigB = np.copy( tree.orbitalEnergies[m] )
#                             if greenIterationsCount>2+aitkenStart: tree.orbitalEnergies[m] = aitkenEig  # keep eig fixed to previous aitken value
                            
                        if (greenIterationsCount-aitkenStart)%4==2:
                            print('Saving psiC')
                            psiC = np.copy(tempOrbital[:,3])
#                             print('sum psiB : ', np.sum(psiB))
                            eigC = np.copy( tree.orbitalEnergies[m] ) 
#                             if greenIterationsCount>2+aitkenStart: tree.orbitalEnergies[m] = aitkenEig  # keep eig fixed to previous aitken value

                            
                            aitkenEig = AitkenAcceleration(eigA, eigB, eigC)
                            aitkenPsi = AitkenAcceleration(psiA, psiB, psiC)
#                             aitkenPsi = AitkenPointwiseAcceleration(psiA,psiB,psiC)
                
#                             aitkenPsi = nathanAcceleration(psiA, psiB, psiC)
#                             aitkenEig = nathanAcceleration(eigA, eigB, eigC)
                            
                            tree.exportAitkenWavefunction(densityPlotsDir+'/aitken_noFilter', tempOrbital[:,0], tempOrbital[:,1], tempOrbital[:,2], psiA, psiB, psiC, aitkenPsi)
                            
                            orbitals[:,m] = np.copy( aitkenPsi )
#                             orthWavefunction = modifiedGramSchmidt_singleOrbital(orbitals,weights,m, n, k)
#                             orbitals[:,m] = np.copy(orthWavefunction)
                            
#                             print('Norm of aitkenPsi: ', np.sqrt( np.sum( aitkenPsi*aitkenPsi*weights ) ))
                            print('Norm diff between aitken and psiA: ', np.sqrt( np.sum( (aitkenPsi-psiA)**2*weights ) ))
                            print('Norm diff between aitken and psiB: ', np.sqrt( np.sum( (aitkenPsi-psiB)**2*weights ) ))
                            print('Norm diff between aitken and psiC: ', np.sqrt( np.sum( (aitkenPsi-psiC)**2*weights ) ))
                            
                            print('Min values of psiA, B, C, Aitken: ', np.min(psiA), np.min(psiB), np.min(psiC), np.min(aitkenPsi) )
                            print('Max values of psiA, B, C, Aitken: ', np.max(psiA), np.max(psiB), np.max(psiC), np.max(aitkenPsi) )
                            
                            idx = np.argmax(aitkenPsi)
                            print('Location of PsiAitken max: ', x[idx], y[idx], z[idx])
                            print('psiA at this location: ', psiA[idx])
                            print('psiB at this location: ', psiB[idx])
                            print('psiC at this location: ', psiC[idx])

                            idx = np.argmin(aitkenPsi)
                            print('Location of PsiAitken min: ', x[idx], y[idx], z[idx])
                            print('psiA at this location: ', psiA[idx])
                            print('psiB at this location: ', psiB[idx])
                            print('psiC at this location: ', psiC[idx])
#                             return
#                             if m==1:
#                                 tempOrbital0 = tree.extractPhi(0)
#                                 print('psiA overlap with psi0:      ', np.sqrt( np.sum( np.abs(psiA*tempOrbital0[:,3])*weights ) ))
#                                 print('psiB overlap with psi0:      ', np.sqrt( np.sum( np.abs(psiB*tempOrbital0[:,3])*weights ) ))
#                                 print('psiC overlap with psi0:      ', np.sqrt( np.sum( np.abs(psiC*tempOrbital0[:,3])*weights ) ))
#                                 print('psiAitken overlap with psi0: ', np.sqrt( np.sum( np.abs(aitkenPsi*tempOrbital0[:,3])*weights ) ))
#                                 epsConv = -0.70603315
#                                 print('EigA - Converged: ', (eigA-epsConv))
#                                 print('EigB - Converged: ', (eigB-epsConv))
#                                 print('EigC - Converged: ', (eigC-epsConv))
#                                 print('EigS - Converged: ', (aitkenEig-epsConv))
#                             
#                             if m==0:   
#                                 epsConv = -4.356131056
#                                 print('EigA - Converged: ', (eigA-epsConv))
#                                 print('EigB - Converged: ', (eigB-epsConv))
#                                 print('EigC - Converged: ', (eigC-epsConv))
#                                 print('EigS - Converged: ', (aitkenEig-epsConv))
                                
                                
#                             aitkenPsi /= np.sqrt( np.sum( aitkenPsi*aitkenPsi*weights ) )  # normalize the aitken wavefunction
                            
#                             # Reset psiOld and eigOld to input values.  Compute update to Aitken values.
#                             tree.importPhiOnLeaves(psiA,m)
#                             tree.orbitalEnergies[m] = eigA
#                             tree.setPhiOldOnLeaves(m) # Sets phiC to be the old psi
#                             tree.importPhiNewOnLeaves(aitkenPsi) # Sets aitkenPsi to be the new psi
#                             tree.updateOrbitalEnergies_NoGradients(m, newOccupations=False)  # computes eigenvalue for transition from psiC to aitkenPsi
                            

                        
                        
                        
                            print('Saving Aitken psi and eigenvalue.')
                            
                            
                            tree.importPhiOnLeaves(orbitals[:,m], m)
                            tree.setPhiOldOnLeaves(m)
                            
                            tree.orbitalEnergies[m] = np.copy(aitkenEig)
#                             eigenvalueDiff = abs(aitkenEig - oldEigenvalue)
                            
           
                            
                            print('Setting aitkenAcceleration back to false.')
                            print('Aitken Eig:                       ', aitkenEig)
                            aitkenAcceleration=False
                            freezeEigenvalue=True


                            
                            
#                             try:
#                                 aitkenNormDiff = np.sqrt( np.sum( (aitkenPsi-oldAitkenPsi)**2*weights ) )
#                                 aitkenEigDiff = abs( aitkenEig - oldAitkenEig )
#                                 
#                                 
#                                 print('Residual of Aitken Wavefunctions: ', aitkenNormDiff)
#                                 print('Residual of Aitken Eigenvalues:   ', aitkenEigDiff)
#                                 
#                                 
# #                                 normDiff = aitkenNormDiff
# #                                 eigenvalueDiff = aitkenEigDiff
#                             except Exception: 
#                                 print('Not computing residual of aitken wavefunction.  This is okay if this is only the second iteration.')
                                
                            
                            oldAitkenPsi=np.copy(aitkenPsi)
#                             oldAitkenEig = np.copy(aitkenEig)
# #                             
                        
                    ##########################################################################################
                    ##########################################################################################

                    else:  # update without aitken
                        tempOrbital = tree.extractPhi(m)
                        orbitals[:,m] = np.copy( tempOrbital[:,3] )
                        if symmetricIteration==False:
#                             print('Computing residual of psi')
                            normDiff = np.sqrt( np.sum( (orbitals[:,m]-oldOrbitals[:,m])**2*weights ) )
                            sumDiff = np.sum((orbitals[:,m]-oldOrbitals[:,m])*weights )
                        elif symmetricIteration==True:
#                             print('Computing residual of psi*sqrtV')
                            normDiff = np.sqrt( np.sum( (orbitals[:,m]*sqrtV-oldOrbitals[:,m]*sqrtV)**2*weights ) )
                        eigenvalueDiff = abs(newEigenvalue - oldEigenvalue)
                    
                    

                    residuals[m] = normDiff
                    orbitalResidual = np.copy(normDiff)
                    
                    
                    if GIandersonMixing==True:
                        
                        if firstOutputWavefunction==True:
                            temp = np.append( orbitals[:,m], tree.orbitalEnergies[m])
                            outputWavefunctions[:,0] = np.copy(temp) # fill first column of outputWavefunctions
#                             outputEigenvalues[0] = tree.orbitalEnergies[m]
                            firstOutputWavefunction=False
                        else:
#                             outputWavefunctions = np.concatenate( ( outputWavefunctions, np.reshape(np.copy(orbitals[:,m]), (numberOfGridpoints,1)) ), axis=1)
                            
                            if (greenIterationsCount-1-mixingStart)<mixingHistoryCutoff:
                                temp = np.append( orbitals[:,m], tree.orbitalEnergies[m])
                                outputWavefunctions = np.concatenate( ( outputWavefunctions, np.reshape(np.copy(temp), (numberOfGridpoints+1,1)) ), axis=1)
#                                 outputEigenvalues = np.append( outputEigenvalues, tree.orbitalEnergies[m])
                                print('Concatenated outputWavefunction.  Now has shape: ', np.shape(outputWavefunctions))
                            else:
                                print('Beyond mixingHistoryCutoff.  Replacing column ', (greenIterationsCount-1-mixingStart)%mixingHistoryCutoff)
#                                 print('Shape of oldOrbitals[:,m]: ', np.shape(oldOrbitals[:,m]))
                                temp = np.append( orbitals[:,m], tree.orbitalEnergies[m])
                                outputWavefunctions[:,(greenIterationsCount-1-mixingStart)%mixingHistoryCutoff] = np.copy(temp)
#                                 outputEigenvalues[(greenIterationsCount-1-mixingStart)%mixingHistoryCutoff] = np.copy(tree.orbitalEnergies[m])
                        
#                         #  Try freezing the eigenvalue after some updates.  Possibly prevent oscillations
#                         if greenIterationsCount%3==0:
#                             print('Freezing eigenvalue for next iteration.')
#                             freezeEigenvalue=True
                        
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
                        andersonOrbital, andersonWeights = densityMixing.computeNewDensity(inputWavefunctions, outputWavefunctions, mixingParameter,np.append(weights,1.0), returnWeights=True)
#                         newEig = densityMixing.applyWeightsToEigenvalue(outputEigenvalues,andersonWeights)
                        
#                         print('Anderson weighted eigenvalue: ', newEig)
#                         tree.orbitalEnergies[m] = newEig

                        # Assuming the orbital occupied first N positions, eigenvalue occupied N+1 position
                        tree.importPhiOnLeaves(andersonOrbital[:-1], m)
                        tree.orbitalEnergies[m] = andersonOrbital[-1]
                        print('Anderson eigenvalue = ', andersonOrbital[-1])
                        
#                         andersonEigenvalue, andersonWeights = densityMixing.computeNewDensity(inputEigenvalues, outputEigenvalues, mixingParameter,weights, returnWeights=True)
                        

                    print('Orbital %i error and eigenvalue residual:   %1.3e and %1.3e' %(m,tree.orbitalEnergies[m]-referenceEigenvalues[m]-tree.gaugeShift, eigenvalueDiff))
                    print('Orbital %i wavefunction residual: %1.3e' %(m, orbitalResidual))
                    print('SumDiff = ', sumDiff)
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
                    
                    
                    residualRatio = orbitalResidual/oldOrbitalResidual
                    eigenvalueResidualRatio = eigenvalueDiff/previousEigenvalueDiff
                    print()
                    print('Wavefunction Relative Residual =          ', residualRatio)
                    print('Wavefunction Previous relative residual = ', previousResidualRatio)
                    print()
                    print('Eigenvalue Relative Residual =          ', eigenvalueResidualRatio)
                    print('Eigenvalue Previous relative residual = ', previousEigenvalueResidualRatio)
                    print()
                    
                    # If wavefunction residual is low then start using Anderson Mixing
                    if ((GIandersonMixing==False)): 
                        try:   
                            psiRatio = abs(residualRatio/previousResidualRatio )
                        except Exception:
                            psiRatio = 0
                        try:
                            eigRatio = abs(eigenvalueResidualRatio/previousEigenvalueResidualRatio )
                        except Exception:
                            eigRatio=0
                        if ( 
                            (abs(1 - psiRatio) < ratioTol) and 
                            (abs(1 - eigRatio) < ratioTol) and 
                            (eigenvalueResidualRatio<1) and
                            (residualRatio<1) 
                             ):
                            if abs(orbitalResidual-oldOrbitalResidual) < abs(oldOrbitalResidual - oldOldOrbitalResidual): # if true, then convergence of residual is slowing down.  Time to accelerate.
                                                                                                                             # otherwise hold off. Maybe residual will continue to fall (if it's falling)
                                print('psiRatio = ', psiRatio)
                                print('eigRatio = ', eigRatio)
                                print('Tolerance: ', ratioTol)
                                GIandersonMixing = True
                                mixingStart = greenIterationsCount
                                print('Turning on Anderson Mixing for wavefunction %i' %m)
                    ### EXIT CONDITIONS ###
 
#                     if SCFcount==1:
#                         print('Loosening Greens iteration tolerance in SCF #1')
#                         orbitalResidual*=1000
#                     if (np.abs(oldOrbitalResidual-orbitalResidual)/np.abs(oldOrbitalResidual) <= 1/10000):
#                         print('Relative residual changing by < 0.01%.  Exiting Greens Iteration')

                    
                    
#                     ratioTol = 1e-1
#                     if ( (aitkenAcceleration==False) and (True==False) ):
#                         try:   
#                             psiRatio = abs(residualRatio/previousResidualRatio )
#                         except Exception:
#                             psiRatio = 1
#                         try:
#                             eigRatio = abs(eigenvalueResidualRatio/previousEigenvalueResidualRatio )
#                         except Exception:
#                             eigRatio=1
#                         if ( 
#                             (abs(1 - psiRatio) < ratioTol) and 
#                             (abs(1 - eigRatio) < ratioTol) and 
#                             (eigenvalueResidualRatio<1) and
#                             (residualRatio<1)
#                              ):
#                             print('Turning on Steffensen acceleration because psi and epsilon are in linear regime.')
#                             aitkenAcceleration=True
#                             freezeEigenvalue=False
#                             aitkenStart = greenIterationsCount
#                             ratioTol = 1e-1


                    if orbitalResidual < intraScfTolerance:
                        print('Used %i iterations for orbital %i.\n\n\n' %(greenIterationsCount,m))
                        
#                     if (abs(eigenvalueDiff) < intraScfTolerance/10000):
#                         print('Eiegnvalue residual smaller than L2 tol/10000, so terminating Green iterations.')
                    
                        
                    
                    if ( (orbitalResidual<1e-2) and (tree.orbitalEnergies[m]>tree.gaugeShift) ):   # if reached 1e-2 residual and energy is positive, this st
                        print('Orbtial residual less than 1e-2 and energy is positive, meaning this state is not resolved.  Hopefully unoccupied.  Terminating Green Iterations.')
                        orbitalResidual=0.0
                        
                    previousResidual = np.copy(orbitalResidual)
                    previousEigenvalueDiff = np.copy(eigenvalueDiff)
                    previousResidualRatio = np.copy(residualRatio)
                    previousEigenvalueResidualRatio = np.copy(eigenvalueResidualRatio)
                    greenIterationsCount += 1 
                    
                
            else:     
                print('orbital %i energy is positive, not updating it anymore.' %m)
        
        
        # sort by energy and compute new occupations
        tree.sortOrbitalsAndEnergies()
        tree.computeOccupations()
        for mm in range(nOrbitals):
            # fill in orbitals  
            targets = tree.extractPhi(mm)
            weights = np.copy(targets[:,5])
            oldOrbitals[:,mm] = np.copy(targets[:,3])
            orbitals[:,mm] = np.copy(targets[:,3])  
#         occupations = computeOccupations(tree.orbitalEnergies, tree.nElectrons, Temperature)
        
        
        ##  DO I HAVE ENOUGH ORBITALS?  CHECK, AND ADD ONE IF NOT.
#         if tree.occupations[-1] > 1e-6:
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
#             outputDensities = np.concatenate( ( outputDensities, np.reshape(np.copy(newDensity), (numberOfGridpoints,1)) ), axis=1)
            
            if (SCFcount-1)<mixingHistoryCutoff:
                outputDensities = np.concatenate( (outputDensities, np.reshape(np.copy(newDensity), (numberOfGridpoints,1))), axis=1)
                print('Concatenated outputDensity.  Now has shape: ', np.shape(outputDensities))
            else:
                print('Beyond mixingHistoryCutoff.  Replacing column ', (SCFcount-1)%mixingHistoryCutoff)
#                                 print('Shape of oldOrbitals[:,m]: ', np.shape(oldOrbitals[:,m]))
                outputDensities[:,(SCFcount-1)%mixingHistoryCutoff] = newDensity
        
        print('Sample of output densities:')
        print(outputDensities[0,:])    
        integratedDensity = np.sum( newDensity*weights )
        densityResidual = np.sqrt( np.sum( (sources[:,3]-oldDensity[:,3])**2*weights ) )
        print('Integrated density: ', integratedDensity)
        print('Density Residual ', densityResidual)
        
#         densityResidual = np.sqrt( np.sum( (outputDensities[:,SCFcount-1] - inputDensities[:,SCFcount-1])**2*weights ) )
#         print('Density Residual from arrays ', densityResidual)
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
#         starthartreeConvolutionTime = timer()

        density_sources = tree.extractLeavesDensity()  # extract the source point locations.  Currently, these are just all the leaf midpoints
        density_targets = np.copy(sources)
        
        if GPUpresent==True:
            if treecode==False:
                V_hartreeNew = np.zeros((len(targets)))
                gpuHartreeGaussianSingularitySubract[blocksPerGrid, threadsPerBlock](targets,density_sources,V_hartreeNew,alphasq)
            elif treecode==True:
                numTargets = len(density_targets)
                numSources = len(density_sources)
                sourceX = np.copy(density_sources[:,0])
    
                sourceY = np.copy(density_sources[:,1])
                sourceZ = np.copy(density_sources[:,2])
                sourceValue = np.copy(density_sources[:,3])
                sourceWeight = np.copy(density_sources[:,4])
                
                targetX = np.copy(density_targets[:,0])
                targetY = np.copy(density_targets[:,1])
                targetZ = np.copy(density_targets[:,2])
                targetValue = np.copy(density_targets[:,3])
                targetWeight = np.copy(density_targets[:,4])
                
                start = time.time()
                potentialType=2 
                alpha = gaussianAlpha
                V_hartreeNew = treecodeWrappers.callTreedriver(numTargets, numSources, 
                                                               targetX, targetY, targetZ, targetValue, 
                                                               sourceX, sourceY, sourceZ, sourceValue, sourceWeight,
                                                               potentialType, alpha, treecodeOrder, theta, maxParNode, batchSize)
                print('Convolution time: ', time.time()-start)
                
        elif GPUpresent==False:
            
            sourceX = np.copy(density_sources[:,0])
            sourceY = np.copy(density_sources[:,1])
            sourceZ = np.copy(density_sources[:,2])
            sourceValue = np.copy(density_sources[:,3])
            sourceWeight = np.copy(density_sources[:,4])
            
            targetX = np.copy(density_targets[:,0])
            targetY = np.copy(density_targets[:,1])
            targetZ = np.copy(density_targets[:,2])
            targetValue = np.copy(density_targets[:,3])
            targetWeight = np.copy(density_targets[:,4])
                
            if treecode==False:
                V_hartreeNew = directSumWrappers.callCompiledC_directSum_PoissonSingularitySubtract(numTargets, numSources, alphasq, 
                                                                                                  targetX, targetY, targetZ, targetValue,targetWeight, 
                                                                                                  sourceX, sourceY, sourceZ, sourceValue, sourceWeight)

                V_hartreeNew += density_targets[:,3]* (4*np.pi)/ alphasq/ 2   # Correct for exp(-r*r/alphasq)  # DONT TRUST

                
            else:
                potentialType=2 # shoud be 0.  Set to 1, 2, or 3 just to test other kernels quickly
                print('NEED TREECODE PARAMS FOR THIS SECTION')
                return
#                 order=3
#                 theta = 0.5
#                 maxParNode = 500
#                 batchSize = 500
#                 alphasq = gaussianAlpha**2
#                 V_hartreeNew = treecodeWrappers.callTreedriver(numTargets, numSources, 
#                                                                targetX, targetY, targetZ, targetValue, 
#                                                                sourceX, sourceY, sourceZ, sourceValue, sourceWeight,
#                                                                potentialType, alphasq, order, theta, maxParNode, batchSize)
#                 if potentialType==2:
#                     V_hartreeNew += density_targets[:,3]* (4*np.pi) / alphasq/2
        
        else:
            print('Is GPUpresent supposed to be true or false?')
            return
      
        tree.importVhartreeOnLeaves(V_hartreeNew)
        tree.updateVxcAndVeffAtQuadpoints()
#         hartreeConvolutionTime = timer() - starthartreeConvolutionTime
#         print('Computing Vhartree and updating Veff took:    %.4f seconds. ' %hartreeConvolutionTime)

        
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
#         print('Updated Kinetic Energy:                 %.10f H, %.10e H' %(tree.totalKinetic, tree.totalKinetic-Ekinetic) )
        print('Updated E_Hartree:                      %.10f H, %.10e H' %(tree.totalEhartree, tree.totalEhartree-Ehartree) )
        print('Updated E_x:                           %.10f H, %.10e H' %(tree.totalEx, tree.totalEx-Eexchange) )
        print('Updated E_c:                           %.10f H, %.10e H' %(tree.totalEc, tree.totalEc-Ecorrelation) )
#         print('Updated totalElectrostatic:            %.10f H, %.10e H' %(tree.totalElectrostatic, tree.totalElectrostatic-Eelectrostatic))
        print('Total Energy:                          %.10f H, %.10e H' %(tree.E, tree.E-Etotal))
        print('Energy Residual:                        %.3e' %energyResidual)
        print('Density Residual:                       %.3e\n\n'%densityResidual)



            
        if vtkExport != False:
            filename = vtkExport + '/mesh%03d'%(SCFcount-1) + '.vtk'
            tree.exportGridpoints(filename)

        printEachIteration=True

        if printEachIteration==True:
            header = ['Iteration', 'densityResidual', 'orbitalEnergies','bandEnergy', 'kineticEnergy', 
                      'exchangeEnergy', 'correlationEnergy', 'hartreeEnergy', 'totalEnergy']
        
            myData = [SCFcount, densityResidual, tree.orbitalEnergies, tree.totalBandEnergy, tree.totalKinetic, 
                      tree.totalEx, tree.totalEc, tree.totalEhartree, tree.E]
            
        
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
            
            sources = tree.extractLeavesDensity()
            np.save(densityFile, sources[:,3])
            np.save(outputDensityFile, outputDensities)
            np.save(inputDensityFile, inputDensities)
            
            np.save(vHartreeFile, V_hartreeNew)
            
            
            
            # make and save dictionary
            auxiliaryRestartData = {}
            auxiliaryRestartData['SCFcount'] = SCFcount
            auxiliaryRestartData['totalIterationCount'] = tree.totalIterationCount
            auxiliaryRestartData['eigenvalues'] = tree.orbitalEnergies
            auxiliaryRestartData['Eold'] = Eold
    
            np.save(auxiliaryFile, auxiliaryRestartData)
        except FileNotFoundError:
            pass
                
        
        if plotSliceOfDensity==True:
#             densitySliceSavefile = densityPlotsDir+'/iteration'+str(SCFcount)
            r, rho = tree.interpolateDensity(xi,yi,zi,xf,yf,zf, numpts, plot=False, save=False)
        
#
            densities = np.load(densitySliceSavefile+'.npy')
            densities = np.concatenate( (densities, np.reshape(rho, (numpts,1))), axis=1)
            np.save(densitySliceSavefile,densities)
    
                
        """ END WRITING INDIVIDUAL ITERATION TO FILE """
     
        
        if tree.E > 0.0:                       # Check that the current guess for energy didn't go positive.  Reset it if it did. 
            print('Warning, Energy is positive')
            tree.E = -0.5
            
        
        if SCFcount >= 150:
            print('Setting density residual to -1 to exit after the 150th SCF')
            densityResidual = -1
            
        if SCFcount >= 1:
            print('Setting density residual to -1 to exit after the First SCF just to test treecode or restart')
            energyResidual = -1
            densityResidual = -1
        


        
    print('\nConvergence to a tolerance of %f took %i iterations' %(interScfTolerance, SCFcount))

