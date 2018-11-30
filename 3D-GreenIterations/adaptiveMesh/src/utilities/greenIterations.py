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
# from convolution import gpuPoissonConvolution,gpuHelmholtzConvolutionSubractSingularity, cpuHelmholtzSingularitySubtract,cpuHelmholtzSingularitySubtract_allNumerical
from convolution import *
import densityMixingSchemes as densityMixing
from networkx.algorithms.flow.utils import CurrentEdge

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

def wavefunctionErrors(wave1, wave2, weights, x,y,z):
    L2error = np.sqrt( np.sum(weights*(wave1-wave2)**2 ) )
    LinfIndex = np.argmax( abs( wave1-wave2 ))
    Linf = wave1[LinfIndex] - wave2[LinfIndex] 
    LinfRel = (wave1[LinfIndex] - wave2[LinfIndex])/wave1[LinfIndex]
    xinf = x[LinfIndex]
    yinf = y[LinfIndex]
    zinf = z[LinfIndex]
    
    print('~~~~~~~~Wavefunction Errors~~~~~~~~~~')
    print("L2 Error:             ", L2error)
    print("Linf Error:           ", Linf)
    print("LinfRel Error:        ", LinfRel)
    print("Located at:           ", xinf, yinf, zinf)
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')



def greenIterations_KohnSham_SCF(tree, intraScfTolerance, interScfTolerance, numberOfTargets, gradientFree, mixingScheme, mixingParameter,
                                subtractSingularity, smoothingN, smoothingEps, inputFile='',outputFile='',
                                onTheFlyRefinement = False, vtkExport=False, outputErrors=False, maxOrbitals=None, maxSCFIterations=None): 
    '''
    Green Iterations for Kohn-Sham DFT using Clenshaw-Curtis quadrature.
    '''
    
#     ## OXYGEN ATOM  
     
    if tree.nOrbitals == 5:
        print('Assuming this is the oxygen atom test case...')  
        dftfeOrbitalEnergies = np.array( [-1.875890295968488530e+01, -8.712069894624057120e-01,
                                          -3.382944529486854868e-01, -3.382944529460307215e-01,
                                          -3.382944529419588120e-01])#, -0.1])   # DFTFE order 5
        tree.occupations = np.array([2,2,4/3,4/3,4/3])
        
    elif tree.nOrbitals == 6:
        print('Assuming this is the oxygen atom test case with one extra orbital...')  
        dftfeOrbitalEnergies = np.array( [-1.875890295968488530e+01, -8.712069894624057120e-01,
                                          -3.382944529486854868e-01, -3.382944529460307215e-01,
                                          -3.382944529419588120e-01, +0.05203574])   # DFTFE order 5
        tree.occupations = np.array([2,2,4/3,4/3,4/3,0])
        
#     ## Carbon Monoxide   
     
    elif ( (tree.nOrbitals == 10) or (tree.nOrbitals == 7) ):
        print('Assuming this is the carbon monoxide test case...')  
        if tree.nOrbitals==10:
            dftfeOrbitalEnergies = np.array( [-1.871914961754098883e+01, -9.907098561099513034e+00,
                                              -1.075307096649917860e+00, -5.215348395483176969e-01,
                                              -4.455546114762743981e-01, -4.455543309534727436e-01,
                                              -3.351421635719918912e-01, -8.279124771292359353e-02,
                                              -8.279124771292359353e-02, 1.290367676602023790e-02]) 
            tree.occupations = np.array([2,2,2,2,2,2,2,0,0,0]) 
        elif tree.nOrbitals==7:
            dftfeOrbitalEnergies = np.array( [-1.871914961754098883e+01, -9.907098561099513034e+00,
                                              -1.075307096649917860e+00, -5.215348395483176969e-01,
                                              -4.455546114762743981e-01, -4.455543309534727436e-01,
                                              -3.351421635719918912e-01]) 
            tree.occupations = np.array([2,2,2,2,2,2,2]) 


    ## BERYLLIUM ATOM     
    elif tree.nOrbitals == 2:
        print('Assuming this is the Beryllium test case...')
        dftfeOrbitalEnergies = np.array( [-3.855615417517944898e+00, -2.059998705089633453e-01 ] ) 
#     dftfeOrbitalEnergies = np.array( [-3.855605920292163535e+00, -2.059995800202487071e-01 ] ) 
        tree.occupations = np.array([2,2])
    
    else:
        print('Not sure which test case this is..., nOrbitals = ', tree.nOrbitals)
        
        return
    
    
    
                                    
    
#     deepestKinetic = 28.75629355  # computed with LW5-4000 order 5, using the initialized orbital.  Total orbital energy agrees to 2.40330518e-05, so each of these pieces should be pretty accurate.
#     deepestPotential = -47.51505323
    
    deepestKinetic = 28.75629355-4.63052339e-05 
    deepestPotential = -47.51505323 + 1.97012921e-05  # thes values obtained with LW5-6000.  The total orbital error was -2.57888546e-06, so should be even better than the LW5-4000 values
     
    greenIterationOutFile = outputFile[:-4]+'_GREEN_'+outputFile[-4:]
    SCFiterationOutFile = outputFile[:-4]+'_SCF_'+outputFile[-4:]
    
    coefficients = np.zeros(smoothingN+1)
    for ii in range(smoothingN+1):
        coefficients[ii] = 1.0
        for jj in range(ii):
            coefficients[ii] /= (jj+1) # this is replacing the 1/factorial(i)
            coefficients[ii] *= ((-1/2)-jj)

    # Initialize density history arrays
    inputDensities = np.zeros((tree.numberOfGridpoints,1))
    outputDensities = np.zeros((tree.numberOfGridpoints,1))
    
    

    threadsPerBlock = 512
    blocksPerGrid = (numberOfTargets + (threadsPerBlock - 1)) // threadsPerBlock  # compute the number of blocks based on N and threadsPerBlock
    
    print('\nEntering greenIterations_KohnSham_SCF()')
    print('\nNumber of targets:   ', numberOfTargets)
    print('Threads per block:   ', threadsPerBlock)
    print('Blocks per grid:     ', blocksPerGrid)
    
    greenIterationCounter=1                                     # initialize the counter to counter the number of iterations required for convergence
    densityResidual = 10                                   # initialize the densityResidual to something that fails the convergence tolerance
    Eold = -0.5 + tree.gaugeShift

#     [Etrue, ExTrue, EcTrue, Eband] = np.genfromtxt(inputFile,dtype=[(str,str,int,int,float,float,float,float,float)])[4:8]
    [Eband, Ekinetic, Eexchange, Ecorrelation, Eelectrostatic, Etotal] = np.genfromtxt(inputFile)[4:10]
    print([Eband, Ekinetic, Eexchange, Ecorrelation, Eelectrostatic, Etotal])

    ### COMPUTE THE INITIAL HAMILTONIAN ###
    targets = tree.extractLeavesDensity()  
    sources = tree.extractLeavesDensity()   # extract density on secondary mesh
#     sources = tree.extractDenstiySecondaryMesh()   # extract density on secondary mesh

    V_coulombNew = np.zeros((len(targets)))
    startCoulombConvolutionTime = timer()
    alpha = 1
    alphasq=alpha*alpha
    
    print('Using Gaussian singularity subtraction, alpha = ', alpha)
    gpuHartreeGaussianSingularitySubract[blocksPerGrid, threadsPerBlock](targets,sources,V_coulombNew,alphasq)
    
    
#     if smoothingN==0:
#     gpuPoissonConvolution[blocksPerGrid, threadsPerBlock](targets,sources,V_coulombNew)  # call the GPU convolution 
#     print('Using singularity subtraction for initial Poisson solve')
#     gpuPoissonConvolutionSingularitySubtract[blocksPerGrid, threadsPerBlock](targets,sources,V_coulombNew,5)
#     else:
#         print('Using smoothed version for Poisson Convolution: (n, epsilon) = (%i, %2.3f)' %(smoothingN, smoothingEps))
#         gpuPoissonConvolutionChristliebSmoothing[blocksPerGrid, threadsPerBlock](targets,sources,V_coulombNew,smoothingN,smoothingEps,coefficients)
    CoulombConvolutionTime = timer() - startCoulombConvolutionTime
    print('Computing Vcoulomb took:    %.4f seconds. ' %CoulombConvolutionTime)
    tree.importVcoulombOnLeaves(V_coulombNew)
    tree.updateVxcAndVeffAtQuadpoints()
    
    #manually set the initial occupations, otherwise oxygen P shell gets none.
#     tree.occupations = np.array([2,2,2,2,2,2,2])
#     tree.occupations = np.array([2,2,4/3,4/3,4/3])
#     tree.occupations = np.array([2,2])
#     tree.occupations = np.array([2,2,2,2,4/3,4/3,4/3])

#     tree.occupations = 2*np.ones(tree.nOrbitals)

#     tree.updateOrbitalEnergies(sortByEnergy=False)
#     print('In the above energies, first 5 are for the carbon, next 5 for the oxygen.')
    print('Update orbital energies after computing the initial Veff.  Save them as the reference values for each cell')
    tree.updateOrbitalEnergies(sortByEnergy=True, saveAsReference=True)
#     print('Kinetic:   ', tree.orbitalKinetic)
#     print('Potential: ', tree.orbitalPotential)
    print('Kinetic Errors:   ', tree.orbitalKinetic - deepestKinetic)
    print('Potential Errors: ', tree.orbitalPotential - deepestPotential - tree.gaugeShift)
    tree.updateTotalEnergy(gradientFree=False)
    """
    
    Print results before SCF 1
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

    print('Orbital Energy Errors after initialization: ', tree.orbitalEnergies-dftfeOrbitalEnergies-tree.gaugeShift)

    print('Updated V_x:                           %.10f Hartree' %tree.totalVx)
    print('Updated V_c:                           %.10f Hartree' %tree.totalVc)
    
    print('Updated Band Energy:                   %.10f H, %.10e H' %(tree.totalBandEnergy, tree.totalBandEnergy-Eband) )
    print('Updated Kinetic Energy:                 %.10f H, %.10e H' %(tree.totalKinetic, tree.totalKinetic-Ekinetic) )
    print('Updated E_x:                           %.10f H, %.10e H' %(tree.totalEx, tree.totalEx-Eexchange) )
    print('Updated E_c:                           %.10f H, %.10e H' %(tree.totalEc, tree.totalEc-Ecorrelation) )
    print('Updated totalElectrostatic:            %.10f H, %.10e H' %(tree.totalElectrostatic, tree.totalElectrostatic-Eelectrostatic))
    print('Total Energy:                          %.10f H, %.10e H' %(tree.E, tree.E-Etotal))
    
    
    print('Computing L2 residual for initialized orbitals...')
    for m in range(tree.nOrbitals):
        tree.computeWavefunctionResidual(m)
    
#     print('Compute residual again after setting eigenvalue to the true value...')
#     tree.orbitalEnergies[0] = dftfeOrbitalEnergies[0]
#     tree.computeWavefunctionResidual(0)
#     return
    
    printInitialEnergies=True

    if printInitialEnergies==True:
        header = ['Iteration', 'densityResidual', 'orbitalEnergies','bandEnergy', 'kineticEnergy', 
                  'exchangeEnergy', 'correlationEnergy', 'electrostaticEnergy', 'totalEnergy']
    
        myData = [0, 10, tree.orbitalEnergies, tree.totalBandEnergy, tree.totalKinetic, 
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
#     return


    
# #     if tree.nOrbitals==7:
#     print('Scrambling valence orbitals')
#     for m in range(4,tree.nOrbitals):
#         tree.scrambleOrbital(m)
#     tree.orbitalEnergies[5] = tree.gaugeShift-0.1

    
    

### CARBON MONOXIDE MOLECULE ###    
#     dftfeOrbitalEnergies = np.array( [-1.871953147002199813e+01, -9.907188115343084078e+00,
#                                       -1.075324514852165958e+00, -5.215419985881135645e-01,
#                                       -4.455527567163568570e-01, -4.455527560478895199e-01,
#                                       -3.351419327004790394e-01, -8.275071966753577701e-02,
#                                       -8.273399296312561324e-02,  7.959071929649078059e-03] )
#  
#     dftfeOrbitalEnergiesFirstSCF = np.array( [-1.886806889771608198e+01, -1.000453177064216703e+01,
#                                       -1.185566512385654470e+00, -6.070957834271104581e-01,
#                                       -5.201957473314797742e-01, -5.201895527962699939e-01,
#                                       -3.959879088687816573e-01, -7.423877195920526584e-02,
#                                       -7.325760563979200057e-02,  1.721054880813185223e-02] )
                                      
#     dftfeOrbitalEnergies = np.array( [-1.871953147002199813e+01, -9.907188115343084078e+00,
#                                       -1.075324514852165958e+00, -5.215419985881135645e-01,
#                                       -4.455527567163568570e-01, -4.455527560478895199e-01,
#                                       -3.351419327004790394e-01])#, -8.275071966753577701e-02,
# #                                      8.273399296312561324e-02,  7.959071929649078059e-03] )

 


       
       
#     dftfeOrbitalEnergiesFirstSCF = np.array( [-1.886806889771608198e+01, -1.000453177064216703e+01,
#                                       -1.185566512385654470e+00, -6.070957834271104581e-01,
#                                       -5.201957473314797742e-01, -5.201895527962699939e-01,
#                                       -3.959879088687816573e-01]) #, -7.423877195920526584e-02,
#                                     #7.325760563979200057e-02,  1.721054880813185223e-02] )
                                      
#     ## OXYGEN ATOM     
#     dftfeOrbitalEnergies = np.array( [-1.875878370505640547e+01, -8.711996463756719322e-01,
#                                       -3.382974161584920147e-01, -3.382974161584920147e-01,
#                                       -3.382974161584920147e-01]) 
    
    ## OXYGEN AFTER FIRST SCF ##
#     dftfeOrbitalEnergies = np.array( [-1.875875893002984895e+01, -8.711960263841991292e-01,
#                                       -3.382940967105963481e-01, -3.382940967105849128e-01,
#                                       -3.382940967105836916e-01])         

    # BERYLLIUM ATOM
#     dftfeOrbitalEnergies = np.array( [0, 0])

    ## H2 Molecule
#     dftfeOrbitalEnergies = np.array( [-7.5499497178953057e-01/2])
                        
    
    

    if vtkExport != False:
#         filename = vtkExport + '/mesh%03d'%(greenIterationCounter-1) + '.vtk'
        filename = vtkExport + '/mesh%03d'%(greenIterationCounter-1) + '.vtk'
        tree.exportGridpoints(filename)
        
    

    initialWaveData = tree.extractPhi(0)
    initialPsi0 = np.copy(initialWaveData[:,3])
    x = np.copy(initialWaveData[:,0])
    y = np.copy(initialWaveData[:,1])
    z = np.copy(initialWaveData[:,2])
    
    
#     print('Scrambling orbitals...')
#     tree.scrambleOrbital(0)
#     tree.scrambleOrbital(1)
#     tree.scrambleOrbital(2)
#     tree.scrambleOrbital(3)
#     tree.scrambleOrbital(4)
#     tree.orbitalEnergies[2]=-1
#     tree.orbitalEnergies[3]=-1
#     tree.orbitalEnergies[4]=-1

    inputIntraSCFtolerance = np.copy(intraScfTolerance)

    
    residuals = 10*np.ones_like(tree.orbitalEnergies)
    SCFcount=0
    while ( densityResidual > interScfTolerance ):
        SCFcount += 1
        print()
        print()
        print('\nSCF Count ', SCFcount)
        if SCFcount > 100:
            return
        
#         if SCFcount == 1:
#             print('setting loose tolerance for initial two SCFs')
#             intraScfTolerance = 1e-3 # setting loose tolerance for initial SCF
#         if SCFcount == 2:
#             intraScfTolerance = inputIntraSCFtolerance  # set it back
#         
        
        # fill in the inputDensities array...
        targets = tree.extractLeavesDensity() 
        if SCFcount==1:
            inputDensities[:,0] = np.copy(targets[:,3])
        else:
            inputDensities = np.concatenate( (inputDensities, np.reshape(targets[:,3], (tree.numberOfGridpoints,1))), axis=1)
            
#             diff = inputDensities[:,SCFcount-1] - andersonDensity
#             if np.max(np.abs(diff)) > 1e-14: print('input density not the same as previously computed anderson density.  Why?')
            
#         print('max of targets[:,3] =          ', max(targets[:,3]))
#         print(inputDensities)
#         print('max of inputDensities column = ', max(inputDensities[:,SCFcount-1]))
        
        
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
            
            inputWavefunctions = np.zeros((tree.numberOfGridpoints,1))
            outputWavefunctions = np.zeros((tree.numberOfGridpoints,1))
#             eigenvalueHistory = np.array(tree.orbitalEnergies[m])
#             print('eigenvalueHistory: ', eigenvalueHistory)

            orbitalResidual = 10
            greenIterationsCount = 1
            max_GreenIterationsCount = 5
            
            tree.orthonormalizeOrbitals(targetOrbital=m)
        
        
            print('Working on orbital %i' %m)
            inputIntraSCFtolerance = np.copy(intraScfTolerance)
            
            while ( ( orbitalResidual > intraScfTolerance ) and ( greenIterationsCount < max_GreenIterationsCount) ):
#             while ( ( orbitalResidual > intraScfTolerance ) ):
            
#                 print('Iteration %i' %greenIterationsCount)
                orbitalResidual = 0.0

                sources = tree.extractPhi(m)
                targets = np.copy(sources)
                weights = np.copy(targets[:,5])
                
                oldOrbitals[:,m] = np.copy(targets[:,3])

                
                if greenIterationsCount==1:
                    inputWavefunctions[:,0] = np.copy(oldOrbitals[:,m]) # fill first column of inputWavefunctions
                else:
                    inputWavefunctions = np.concatenate( ( inputWavefunctions, np.reshape(np.copy(oldOrbitals[:,m]), (tree.numberOfGridpoints,1)) ), axis=1)
                


                sources = tree.extractGreenIterationIntegrand(m)
                targets = np.copy(sources)

                if tree.orbitalEnergies[m] > 0:
#                 if tree.orbitalEnergies[m] > tree.gaugeShift:
                    print('Wanrning: Orbital energy greater than gauge shift.')
#                     print('Resetting orbital %i energy to 1/2 gauge shift')
#                     tree.orbitalEnergies[m] = tree.gaugeShift
                    tree.orbitalEnergies[m] = tree.gaugeShift
#                     print('Setting orbital %i energy to zero' %m)

                    
                k = np.sqrt(-2*tree.orbitalEnergies[m])
                phiNew = np.zeros((len(targets)))
#                 if greenIterationsCount<10:
#                 if orbitalResidual > -1e10:
                if subtractSingularity==0: 
#                     print('Using singularity skipping')
                    gpuHelmholtzConvolution[blocksPerGrid, threadsPerBlock](targets,sources,phiNew,k) 
#                 else:
                elif subtractSingularity==1:
                    if tree.orbitalEnergies[m] < -0.2:
                        print('Using singularity subtraction')
#                         gpuHelmholtzConvolutionSubractSingularity_multDivbyr[blocksPerGrid, threadsPerBlock](targets,sources,phiNew,k) 
                        gpuHelmholtzConvolutionSubractSingularity[blocksPerGrid, threadsPerBlock](targets,sources,phiNew,k) 
#                         gpuHelmholtzConvolution[blocksPerGrid, threadsPerBlock](targets,sources,phiNew,k)
#                         gpuHelmholtzConvolutionSubractSingularity_gaussian[blocksPerGrid, threadsPerBlock](targets,sources,phiNew,k,1)
#                         print('Using singularity subtraction with Gaussian, a=0.1')
#                         gpuHelmholtzConvolutionSubractSingularity_gaussian_no_cusp[blocksPerGrid, threadsPerBlock](targets,sources,phiNew,k,0.1)
                    else:
                        print('Using singularity skipping because energy too close to 0')
                        gpuHelmholtzConvolution[blocksPerGrid, threadsPerBlock](targets,sources,phiNew,k)
                else:
                    print('Invalid option for singularitySubtraction, should be 0 or 1.')
                    return
                
                """ Method where you dont compute kinetics, from Harrison """
                
                # update the energy first
                tree.importPhiNewOnLeaves(phiNew)
                
                if ( (gradientFree==True) and (SCFcount>1) ):
                    
                    # orthonormalize before energy update (unsure if I should do this)
                    tree.orthonormalizeOrbitals(targetOrbital=m)
                    
                    
#                     storedEnergies = np.copy(tree.orbitalEnergies)
#                     print('First compute eigenvalue with gradients, just to compare.')
#                     tree.updateOrbitalEnergies(sortByEnergy=False, targetEnergy=m)
#                     gradientEnergy = tree.orbitalEnergies[m]
#                     tree.orbitalEnergies = np.copy(storedEnergies)
                    
                        
                    tree.updateOrbitalEnergies_NoGradients(m, newOccupations=False)
                    if greenIterationsCount==1:
                        eigenvalueHistory = np.array(tree.orbitalEnergies[m])
                    else:
                        eigenvalueHistory = np.append(eigenvalueHistory, tree.orbitalEnergies[m])
                    print('eigenvalueHistory: ',eigenvalueHistory)
                    
                    
                    print('Orbital energy after Harrison update: ', tree.orbitalEnergies[m])
                    if tree.orbitalEnergies[m]>0:
                        tree.orbitalEnergies[m] = -1
#                         print('setting eigenvalue to -1 because it was positive')
                        tree.orbitalEnergies[m] = tree.gaugeShift-0.5
                        print('setting eigenvalue to gauge shift minus 0.5 because it was positive')
#                     gradientFreeEnergy = tree.orbitalEnergies[m]
#                     print('Gradient Energy:      ', gradientEnergy)
#                     print('Gradient-free energy: ', gradientFreeEnergy)
#                     print('Discrepancy:          ', gradientEnergy-gradientFreeEnergy)
#                     print('stored orbital energy: ', tree.orbitalEnergies[m])
                    
                elif ( (gradientFree==False) or (SCFcount==1) ):
#                     # first compute without gradient, for comparison
#                     storedEnergies = np.copy(tree.orbitalEnergies)
#                     print('First compute eigenvalue without gradients, just to compare.')
#                     tree.updateOrbitalEnergies_NoGradients(m, newOccupations=False)
#                     gradientFreeEnergy = tree.orbitalEnergies[m]
#                     tree.orbitalEnergies = np.copy(storedEnergies)
                    
                    tree.updateOrbitalEnergies(sortByEnergy=False, targetEnergy=m)
#                     gradientEnergy = tree.orbitalEnergies[m]
#                     print('Orbital energy after gradient update: ', tree.orbitalEnergies[m])
#                     
#                     print('Gradient Energy:      ', gradientEnergy)
#                     print('Gradient-free energy: ', gradientFreeEnergy)
#                     print('Discrepancy:          ', gradientEnergy-gradientFreeEnergy)
#                     print('stored orbital energy: ', tree.orbitalEnergies[m])
                    
                    
                else:
                    print('Invalid option for gradientFree, which is set to: ', gradientFree)
                    print('type: ', type(gradientFree))

                # now update the orbital
                orbitals[:,m] = np.copy(phiNew)
                tree.importPhiOnLeaves(orbitals[:,m], m)
                tree.orthonormalizeOrbitals(targetOrbital=m)
                
                
                
                
                # Compute norms
                tempOrbital = tree.extractPhi(m)
                orbitals[:,m] = tempOrbital[:,3]
                normDiff = np.sqrt( np.sum( (orbitals[:,m]-oldOrbitals[:,m])**2*weights ) )
                
                if greenIterationsCount==1:
                    outputWavefunctions[:,0] = np.copy(orbitals[:,m]) # fill first column of outputWavefunctions
                else:
                    outputWavefunctions = np.concatenate( ( outputWavefunctions, np.reshape(np.copy(orbitals[:,m]), (tree.numberOfGridpoints,1)) ), axis=1)
                    
                    
                
                GIandersonMixing=False
                GIsimpleMixing=False
                
                if GIsimpleMixing==True:
                    print('Simple mixing on the orbital.')
                    simplyMixedOrbital =  (1-mixingParameter)*orbitals[:,m]+ mixingParameter*oldOrbitals[:,m] 
                    tree.importPhiOnLeaves(simplyMixedOrbital, m)
                    
                if GIandersonMixing==True:
                    print('Anderson mixing on the orbital.')
                    andersonOrbital, andersonWeights = densityMixing.computeNewDensity(inputWavefunctions, outputWavefunctions, mixingParameter,weights, returnWeights=True)
                    tree.importPhiOnLeaves(andersonOrbital, m)
                    
                    
# #                     print('shape of eigenvalueHistory: ', np.shape(eigenvalueHistory))
#                     print(eigenvalueHistory)
# #                     n = len(np.array(eigenvalueHistory))
#                     n = greenIterationsCount
# #                     print('n = ', n)
#                     if greenIterationsCount==1:
#                         currentEnergy = eigenvalueHistory
#                     else:
#                         currentEnergy = np.copy(eigenvalueHistory[n-1])
#                     newEnergy = np.copy(currentEnergy)
# #                     print('Current energy: ', currentEnergy)
# #                     print('Anderson Weights: ', andersonWeights)
# #                     print('Eig History:      ', eigenvalueHistory)
#                     for k in range(n-1):
# #                         print('\nk = ', k)
# #                         print('weight: ', andersonWeights[k])
# #                         print('eig:    ', eigenvalueHistory[n-2-k])
#                         newEnergy += andersonWeights[k] * (eigenvalueHistory[n-2-k] - currentEnergy)
# #                         print('current Energy: ', currentEnergy)
# #                     print('New energy: ', newEnergy)
#                     tree.orbitalEnergies[m] = newEnergy 
# #                     eigenvalueHistory = np.append(eigenvalueHistory, tree.orbitalEnergies[m])
# #                     print('eigenvalueHistory: ',eigenvalueHistory)
                    
                residuals[m] = normDiff
                if normDiff > orbitalResidual:
                    orbitalResidual = np.copy(normDiff) 


                """  Method where you compute kinetics explicitly
#                 idx = np.argmax(phiNew)
#                 phiAtBoundary=phiNew[idx]
#                 print('Phi at boundary: ', phiAtBoundary,'. Shifting wavefunction down by that amount before computing energy.')
#                 print('Location: ', targets[idx][0:3])
#                 phiNew -= phiAtBoundary
                
                orbitals[:,m] = np.copy(phiNew)

                # normalize
#                 print('Not normalizing...')
#                 orbitals[:,m] /= np.sqrt( np.dot(orbitals[:,m],orbitals[:,m]*weights) )
#                 normalizedOrbitals = normalizeOrbitals(orbitals,weights)  # could optimize this to only normalize the current orbital
                
            
                tree.importPhiOnLeaves(orbitals[:,m], m)
#                 tree.updateOrbitalEnergies(sortByEnergy=False, targetEnergy=m)
#                 print('Before orthonormalization:  Orbital %i error:        %1.3e' %(m, tree.orbitalEnergies[m]-dftfeOrbitalEnergies[m]-tree.gaugeShift))
#                 tree.computeWavefunctionResidual(m)
#                 print('Orthonormalizing.')
                tree.orthonormalizeOrbitals(targetOrbital=m)
                tempOrbital = tree.extractPhi(m)
                orbitals[:,m] = tempOrbital[:,3]
                normDiff = np.sqrt( np.sum( (orbitals[:,m]-oldOrbitals[:,m])**2*weights ) )
#                 print('Residual for orbital %i: %1.5e' %(m,normDiff))
                residuals[m] = normDiff
                if normDiff > orbitalResidual:
                    orbitalResidual = np.copy(normDiff) 
                    
                    
#                 oldEnergy = tree.orbitalEnergies[0]
                tree.updateOrbitalEnergies(sortByEnergy=False, targetEnergy=m)
#                 tree.orbitalEnergies[0] = oldEnergy
#                 print('NOT UPDATING ORBITAL ENERGY')
                """
                
#                 print('Computing L2 residual for orbitals after Green Iterations...')
#                 tree.computeWavefunctionResidual(m)
#                 print('Compute residual again after setting eigenvalue to the true value...')
#                 tree.orbitalEnergies[0] = dftfeOrbitalEnergies[0]
#                 tree.computeWavefunctionResidual(0)
                
            
#                 if maxOrbitals==1:
                if m==0:
                    tree.compareToReferenceEnergies()
    #                 print('Orbital %i kinetic and potential: %2.8e and %2.8e ' %(m,tree.orbitalKinetic[m], tree.orbitalPotential[m]-tree.gaugeShift ))
    #                 print('Orbital %i kinetic and potential errors: %2.8e and %2.8e ' %(m,tree.orbitalKinetic[m] - deepestKinetic, tree.orbitalPotential[m]-tree.gaugeShift - deepestPotential ))
                    print("~~~~~~~~~Eigenvalue Errors~~~~~~~~~~~")
#                     print('kinetic error:             %2.8e' %(tree.orbitalKinetic[m] - deepestKinetic) )
#                     print('potential errors:          %2.8e' %(tree.orbitalPotential[m]-tree.gaugeShift - deepestPotential ))
                    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
                    wavefunctionErrors(orbitals[:,m], initialPsi0, weights, x, y, z)
                print('Orbital %i error and residual:   %1.3e and %1.3e' %(m, tree.orbitalEnergies[m]-dftfeOrbitalEnergies[m]-tree.gaugeShift, orbitalResidual))
                print()
                print()

  

    #             SCFiterationOutFile = '/home/njvaughn/CO_LW3_800_SCF1_skipping_minD3.csv'
#                 greenIterationOutFile = '/home/njvaughn/CO_LW3_400_SCF1.csv'
#                 greenIterationOutFile = '/home/njvaughn/CO_LW3_400_SCF1_subtract.csv'
#                 greenIterationOutFile = '/home/njvaughn/iterationResults/CO_LW3_400_SCF_atomicCore_7orb.csv'
#                 greenIterationOutFile = '/home/njvaughn/CO_LW3_1200_skip_GreenIterations_tighter.csv'
    #             SCFiterationOutFile = '/home/njvaughn/CO_LW3_1200_SCF1_skipping.csv'
                header = ['targetOrbital', 'Iteration', 'orbitalResiduals', 'energyEigenvalues']
#                 header = ['targetOrbital', 'Iteration', 'orbitalResiduals', 'energyEigenvalues', 'energyErrors']
        
                myData = [m, greenIterationsCount, residuals,
                          tree.orbitalEnergies-tree.gaugeShift]
#                 myData = [m, greenIterationsCount, residuals,
#                           tree.orbitalEnergies-tree.gaugeShift,
#                           tree.orbitalEnergies-dftfeOrbitalEnergies-tree.gaugeShift]
#                 myData = [m, greenIterationsCount, residuals,
#                           tree.orbitalEnergies-tree.gaugeShift]
                
        
                if not os.path.isfile(greenIterationOutFile):
                    myFile = open(greenIterationOutFile, 'a')
                    with myFile:
                        writer = csv.writer(myFile)
                        writer.writerow(header) 
                    
                
                myFile = open(greenIterationOutFile, 'a')
                with myFile:
                    writer = csv.writer(myFile)
                    writer.writerow(myData)
                
                
                if orbitalResidual < intraScfTolerance:
                    print('Used %i iterations for orbital %i.' %(greenIterationsCount,m))
                greenIterationsCount += 1
        
        
        # sort by energy and compute new occupations
        tree.sortOrbitalsAndEnergies()
        tree.computeOccupations()
#         tree.computeOrbitalMoments()
            
            

 

        print()
        print()
#         print('Computing L2 residual for orbitals after Green Iterations...')
#         for m in range(tree.nOrbitals):
#             tree.computeWavefunctionResidual(m)
        
    #     print('Compute residual again after setting eigenvalue to the true value...')
    #     tree.orbitalEnergies[0] = dftfeOrbitalEnergies[0]
    #     tree.computeWavefunctionResidual(0)
#         return

        

        
        if maxOrbitals==1:
            print('Not updating density or anything since only computing one of the orbitals, not all.')
            return
        
#         tree.orthonormalizeOrbitals()
#         print('Not updating orbital energies ')
#         tree.updateOrbitalEnergies(sortByEnergy=True)
        oldDensity = tree.extractLeavesDensity()
#         print('NOT UPDATING DENSITY AT END OF FIRST SCF.  JUST COMPUTING ENERGIES.')
        
        
        
        tree.updateDensityAtQuadpoints()
         
        sources = tree.extractLeavesDensity()  # extract the source point locations.  Currently, these are just all the leaf midpoints
        targets = np.copy(sources)
        newDensity = np.copy(sources[:,3])
        
        if SCFcount==1:
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
#         print('top row of input: \n', inputDensities[0,:])
#         print('top row of output: \n', outputDensities[0,:])
        
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
            
#             sources = tree.extractLeavesDensity()  # extract the source point locations.  Currently, these are just all the leaf midpoints
#             targets = np.copy(sources)
#             newDensity = sources[:,3]
#             integratedDensity = np.sum( newDensity*weights )
#             densityResidual = np.sqrt( np.sum( (sources[:,3]-oldDensity[:,3])**2*weights ) )
#             print('Integrated density: ', integratedDensity)
#             print('Density Residual ', densityResidual)
#             if SCFcount==1:
#                 # simple mixing
#                 print('Using simple mixing after SCF 1')
#                 simpleMixingDensity = mixingParameter*outputDensities[:,0] + (1-mixingParameter)*inputDensities[:,0]
#                 tree.importDensityOnLeaves(simpleMixingDensity)
#             elif SCFcount==2:
#                 # simple mixing
#                 print('Using simple mixing after SCF 2')
#                 simpleMixingDensity = mixingParameter*outputDensities[:,1] + (1-mixingParameter)*inputDensities[:,1]
#                 tree.importDensityOnLeaves(simpleMixingDensity)
#             elif SCFcount>2:
#                 tree.importDensityOnLeaves(andersonDensity)
 
        """ 
        Compute new electron-electron potential and update pointwise potential values 
        """
        startCoulombConvolutionTime = timer()
        sources = tree.extractLeavesDensity()  # extract the source point locations.  Currently, these are just all the leaf midpoints
        targets = np.copy(sources)
        V_coulombNew = np.zeros((len(targets)))
        gpuHartreeGaussianSingularitySubract[blocksPerGrid, threadsPerBlock](targets,sources,V_coulombNew,alphasq)
        
        
#         if smoothingN==0:
#             gpuPoissonConvolution[blocksPerGrid, threadsPerBlock](targets,sources,V_coulombNew)  # call the GPU convolution 
# #         print('Using singularity subtraction for the Poisson solve!')
# #         gpuPoissonConvolutionSingularitySubtract[blocksPerGrid, threadsPerBlock](targets,sources,V_coulombNew,5)  # call the GPU convolution 
#         else:
#             print('Using smoothed version for Poisson Convolution: (n, epsilon) = (%i, %2.3f)' %(smoothingN, smoothingEps))
#             gpuPoissonConvolutionChristliebSmoothing[blocksPerGrid, threadsPerBlock](targets,sources,V_coulombNew,smoothingN,smoothingEps,coefficients)
        
        tree.importVcoulombOnLeaves(V_coulombNew)
        tree.updateVxcAndVeffAtQuadpoints()
        CoulombConvolutionTime = timer() - startCoulombConvolutionTime
        print('Computing Vcoulomb and updating Veff took:    %.4f seconds. ' %CoulombConvolutionTime)

        """ 
        Compute the new orbital and total energies 
        """
#         startEnergyTime = timer()
#         tree.updateTotalEnergy()
#         print('Band energies before Veff update: %1.6f H, %1.2e H'
#               %(tree.totalBandEnergy, tree.totalBandEnergy-Eband))
#         print('Total Energy without computing orbitals energies with new Veff: %.10f H, %.10e H' %(tree.E, tree.E-Etotal))
#         print('Orbital energies before recomputing with new Veff: ', tree.orbitalEnergies)
#         print('Now recompute orbital energies with new Veff...')


#         tree.updateOrbitalEnergies(sortByEnergy=False) # should I be doing this?
        
#         tree.updateOrbitalEnergies(correctPositiveEnergies=False, sortByEnergy=False) 
        tree.updateTotalEnergy(gradientFree=gradientFree) 
        print('Band energies after Veff update: %1.6f H, %1.2e H'
              %(tree.totalBandEnergy, tree.totalBandEnergy-Eband))
        print('Orbital Energy Errors after Veff Update: ', tree.orbitalEnergies-dftfeOrbitalEnergies-tree.gaugeShift)
        
#         print('Using DFT-FE results from after first SCF...')
        for m in range(tree.nOrbitals):
#             print('Orbital %i error: %1.3e' %(m, tree.orbitalEnergies[m]-dftfeOrbitalEnergiesFirstSCF[m]-tree.gaugeShift))
            print('Orbital %i error: %1.3e' %(m, tree.orbitalEnergies[m]-dftfeOrbitalEnergies[m]-tree.gaugeShift))
        
        
        energyResidual = abs( tree.E - Eold )  # Compute the densityResidual for determining convergence
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

#         if vtkExport != False:
#             tree.exportGreenIterationOrbital(vtkExport,greenIterationCounter)
            
        if vtkExport != False:
            filename = vtkExport + '/mesh%03d'%(greenIterationCounter-1) + '.vtk'
            tree.exportGridpoints(filename)

        printEachIteration=True

        if printEachIteration==True:
            header = ['Iteration', 'densityResidual', 'orbitalEnergies','bandEnergy', 'kineticEnergy', 
                      'exchangeEnergy', 'correlationEnergy', 'electrostaticEnergy', 'totalEnergy']
        
            myData = [greenIterationCounter, densityResidual, tree.orbitalEnergies, tree.totalBandEnergy, tree.totalKinetic, 
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
        
        if greenIterationCounter >= 150:
            print('Setting density residual to -1 to exit after the 150th SCF')
            densityResidual = -1
        
#         print('Setting density residual to min of density residual and energy residual, just for testing purposes')
#         densityResidual = min(densityResidual, energyResidual)
        
        if maxSCFIterations == 1:
            print('Setting density residual to -1 to exit after the first SCF')
            densityResidual = -1 
    
    

        
    print('\nConvergence to a tolerance of %f took %i iterations' %(interScfTolerance, greenIterationCounter))

# def greenIterations_KohnSham_SCF_parallelOrbitals(tree, intraScfTolerance, interScfTolerance, numberOfTargets, 
#                                 subtractSingularity, smoothingN, smoothingEps, inputFile='',SCFiterationOutFile='iterationConvergence.csv',
#                                 onTheFlyRefinement = False, vtkExport=False, outputErrors=False): 
#     '''
#     Green Iterations for Kohn-Sham DFT using Clenshaw-Curtis quadrature.
#     '''
#     threadsPerBlock = 512
#     blocksPerGrid = (numberOfTargets + (threadsPerBlock - 1)) // threadsPerBlock  # compute the number of blocks based on N and threadsPerBlock
#     
#     print('\nEntering greenIterations_KohnSham_SCF()')
#     print('\nNumber of targets:   ', numberOfTargets)
#     print('Threads per block:   ', threadsPerBlock)
#     print('Blocks per grid:     ', blocksPerGrid)
#     
#     greenIterationCounter=1                                     # initialize the counter to counter the number of iterations required for convergence
#     densityResidual = 1                                    # initialize the densityResidual to something that fails the convergence tolerance
#     Eold = -0.5 + tree.gaugeShift
# 
# #     [Etrue, ExTrue, EcTrue, Eband] = np.genfromtxt(inputFile,dtype=[(str,str,int,int,float,float,float,float,float)])[4:8]
#     [Etrue, ExTrue, EcTrue, Eband] = np.genfromtxt(inputFile)[4:8]
#     print([Etrue, ExTrue, EcTrue, Eband])
# 
#     ### COMPUTE THE INITIAL HAMILTONIAN ###
#     targets = tree.extractLeavesDensity()  
#     sources = tree.extractLeavesDensity() 
# 
#     V_coulombNew = np.zeros((len(targets)))
#     gpuPoissonConvolution[blocksPerGrid, threadsPerBlock](targets,sources,V_coulombNew)  # call the GPU convolution 
#     tree.importVcoulombOnLeaves(V_coulombNew)
#     tree.updateVxcAndVeffAtQuadpoints()
#     
#     #manually set the initial occupations, otherwise oxygen P shell gets none.
#     tree.occupations = np.array([2,2,1/3,1/3,1/3,2,2,2/3,2/3,2/3])
# #     tree.updateOrbitalEnergies(sortByEnergy=False)
# #     print('In the above energies, first 5 are for the carbon, next 5 for the oxygen.')
#     print('Update orbital energies after computing the initial Veff.')
#     tree.updateOrbitalEnergies(sortByEnergy=True)
#     
# #     print('Printing orbital energies directly from tree, not from the update function...')
# #     print(tree.orbitalEnergies)
# #     print('Now update orbital energies again.  No sorting should be necessary.')
# #     tree.updateOrbitalEnergies(sortByEnergy=True)
#     
# ### CARBON MONOXIDE MOLECULE ###    
#     dftfeOrbitalEnergies = np.array( [-1.871953147002199813e+01, -9.907188115343084078e+00,
#                                       -1.075324514852165958e+00, -5.215419985881135645e-01,
#                                       -4.455527567163568570e-01, -4.455527560478895199e-01,
#                                       -3.351419327004790394e-01, -8.275071966753577701e-02,
#                                       -8.273399296312561324e-02,  7.959071929649078059e-03] )
#     
#     
#     dftfeOrbitalEnergiesFirstSCF = np.array( [-1.886806889771608198e+01, -1.000453177064216703e+01,
#                                       -1.185566512385654470e+00, -6.070957834271104581e-01,
#                                       -5.201957473314797742e-01, -5.201895527962699939e-01,
#                                       -3.959879088687816573e-01]) #, -7.423877195920526584e-02,
#                                       #-7.325760563979200057e-02,  1.721054880813185223e-02] )
#     
#     
# 
# ### OXYGEN ATOM ###
# #     dftfeOrbitalEnergies = np.array( [-1.875878370505640547e+01, -8.711996463756719322e-01,
# #                                       -3.382974161584920147e-01, -3.382974161584920147e-01,
# #                                       -3.382974161584920147e-01] )
#     
#     
#     
#     if vtkExport != False:
# #         filename = vtkExport + '/mesh%03d'%(greenIterationCounter-1) + '.vtk'
#         filename = vtkExport + '/mesh%03d'%(greenIterationCounter-1) + '.vtk'
#         tree.exportGridpoints(filename)
#         
#     
#     oldOrbitalEnergies = 10
# #     tree.nOrbitals = 5
# #     tree.occupations = [2,2,2,2,2]
#     while ( densityResidual > interScfTolerance ):
#         print()
#         print()
#         print('\nSCF Count ', greenIterationCounter)
#         
# #         ###
# #         tree.orthonormalizeOrbitals()
# #         tree.updateDensityAtQuadpoints()
# #         tree.normalizeDensity()
# #         sources = tree.extractLeavesDensity()  # extract the source point locations.  Currently, these are just all the leaf midpoints
# #         targets = np.copy(sources)
# # 
# #         """ 
# #         Compute new electron-electron potential and update pointwise potential values 
# #         """
# #         startCoulombConvolutionTime = timer()
# #         V_coulombNew = np.zeros((len(targets)))
# #         gpuPoissonConvolution[blocksPerGrid, threadsPerBlock](targets,sources,V_coulombNew)  # call the GPU convolution 
# #         ###gpuPoissonConvolutionSingularitySubtract[blocksPerGrid, threadsPerBlock](targets,sources,V_coulombNew,5)  # call the GPU convolution 
# #         tree.importVcoulombOnLeaves(V_coulombNew)
# #         tree.updateVxcAndVeffAtQuadpoints()
# #         CoulombConvolutionTime = timer() - startCoulombConvolutionTime
# #         print('Computing Vcoulomb and updating Veff took:    %.4f seconds. ' %CoulombConvolutionTime)
# #         ###
# 
#         orbitalResidual = 10
#         greenIterationsCount = 0
#         max_GreenIterationsCount = 555
#         
#         
#         while ( ( orbitalResidual > intraScfTolerance ) and ( greenIterationsCount < max_GreenIterationsCount) ):
#             
# #             print('Setting orbital energies to the DFT-FE values')
# #             tree.orbitalEnergies = dftfeOrbitalEnergiesFirstSCF
# #             
#             orbitalResidual = 0.0
#             
#             """ N ORBITAL HELMHOLTZ SOLVES """
#             orbitals = np.zeros((len(targets),tree.nOrbitals))
#             oldOrbitals = np.zeros((len(targets),tree.nOrbitals))
#             if greenIterationsCount<999:
#                 print('Using singularity skipping...')
#             else:
#                 print('Using singularity subtraction...')
#             for m in range(tree.nOrbitals):
# #                 print('Convolving orbital %i' %m)
#                 sources = tree.extractPhi(m)
#                 targets = np.copy(sources)
#                 weights = np.copy(targets[:,5])
#                 
#                 oldOrbitals[:,m] = np.copy(targets[:,3])
# #                 if ( (m==1) and (greenIterationCounter<-1)):
# #                     print('Not computing new phi1')
# #                     orbitals[:,m] = np.copy(targets[:,3])
# #                 else:
# #                     weights = np.copy(targets[:,5])
#     #                 phiNew = np.zeros((len(targets)))
#     
# #                 k = np.sqrt(-2*tree.orbitalEnergies[m])
# #                     if tree.orbitalEnergies[m] < tree.gaugeShift:
# #                         k = np.sqrt(-2*tree.orbitalEnergies[m])
# #                     else:
# #                         temporaryEpsilon = tree.gaugeShift-1/(m+1)
# #                         k = np.sqrt(-2*temporaryEpsilon)
# #                         print('Orbital %i energy %1.3e > Gauge Shift. Resetting to %1.3f' 
# #                               %(m,tree.orbitalEnergies[m],temporaryEpsilon))
#                     
#     
# #                     phiT=np.copy(targets[:,3])
# #                     vT = np.copy(targets[:,4])
# #                     analyticPiece = -2*phiT*vT/k**2
# #                      
# #                  
# #     # #                 print('max abs of analytic piece: ', max(abs(analyticPiece)))
# #                     minIdx = np.argmin(analyticPiece)  
# #                     maxIdx = np.argmax(analyticPiece) 
# #                     print('~'*50)
# #                     print('Orbital %i, k=%1.3f' %(m,k))
# #                     print('1/(r*epsilon) near nucleus: ', 1/tree.rmin/tree.orbitalEnergies[m])
# #                     print('min analytic piece: ', analyticPiece[minIdx])
# #                     rmin = np.sqrt(targets[minIdx,0]**2 + targets[minIdx,1]**2 + targets[minIdx,2]**2 )
# #                     print('min occurred at r = ', rmin)
# #                     print()
# #                     rmax = np.sqrt(targets[maxIdx,0]**2 + targets[maxIdx,1]**2 + targets[maxIdx,2]**2 )
# #                     print('max analytic piece: ', analyticPiece[maxIdx])
# #                     print('max occurred at r = ', rmax)
# #                     print('~'*50)
#                 
#                 if ( (greenIterationCounter>1) and (tree.occupations[m] < 1e-20)):
#                     print('Not updating phi%i'%m)
#                     orbitals[:,m] = oldOrbitals[:,m]
#                 else:
#                 
# #                 if tree.occupations[m] > 1e-20:  # don't compute new orbitals if it is not occupied.
#                     k = np.sqrt(-2*tree.orbitalEnergies[m])
#                     phiNew = np.zeros((len(targets)))
#                     if greenIterationsCount<999:
#                         print('Using singularity skipping')
#                         gpuHelmholtzConvolution[blocksPerGrid, threadsPerBlock](targets,sources,phiNew,k) 
#                     else:
#                         print('Using singularity subtraction')
#                         gpuHelmholtzConvolutionSubractSingularity[blocksPerGrid, threadsPerBlock](targets,sources,phiNew,k) 
#     #                     k2 = 5
#     #                     gpuHelmholtzConvolutionSubractSingularity_k2[blocksPerGrid, threadsPerBlock](targets,sources,phiNew,k,k2) 
#                 
#                     orbitals[:,m] = np.copy(phiNew)
# #                 else:
# #                     print('Not updating phi%i' %m)
# #                     orbitals[:,m] = oldOrbitals[:,m]
# 
# #                     minIdxIn = np.argmin(sources[:,3])  
# #                     maxIdxIn = np.argmax(sources[:,3]) 
# #                     minIdxOut = np.argmin(phiNew)  
# #                     maxIdxOut = np.argmax(phiNew) 
# #                     
# #                     print('phi%i'%m)
# #                     print('input min:  ', sources[minIdxIn,3])
# #                     print('output min: ', phiNew[minIdxOut])
# #                     print('input max:  ', sources[maxIdxIn,3])
# #                     print('output max: ', phiNew[maxIdxOut])
# #                     print('input min occurred at x,y,z = ', sources[minIdxIn,0:3])
# #                     print('input max occurred at x,y,z = ', sources[maxIdxIn,0:3])
# #                     print('output min occurred at x,y,z = ', sources[minIdxOut,0:3])
# #                     print('output max occurred at x,y,z = ', sources[maxIdxOut,0:3])
# 
# 
# ### Plan:      1. Normalize the freshly computed orbitals. 
# ###            2. Import them into the tree.  
# ###            3. Compute energies, sorting orbitals where needed
# ###            4. Orthonormalize
# ###            5. Compute new energiesm resort when necessary
#             residuals = np.zeros_like(tree.orbitalEnergies)
#             normalizedOrbitals = normalizeOrbitals(orbitals,weights)
#             for m in range(tree.nOrbitals):
#                 tree.importPhiOnLeaves(normalizedOrbitals[:,m], m)
#                 normDiff = np.sqrt( np.sum( (normalizedOrbitals[:,m]-oldOrbitals[:,m])**2*weights ) )
#                 print('Residual for orbital %i: %1.5e' %(m,normDiff))
#                 residuals[m] = normDiff
#                 if normDiff > orbitalResidual:
#                     orbitalResidual = np.copy(normDiff) 
#             print('Computing orbital energies before orthogonalizing.  Sort when needed.')
# # #             tree.updateOrbitalEnergies(sortByEnergy=False)
#             tree.updateOrbitalEnergies(sortByEnergy=True)
# #             print('Using DFT-FE results from after first SCF (before orthogonalization) ...')
# #             for m in range(tree.nOrbitals):
# #                 print('Orbital %i error: %1.3e' %(m, tree.orbitalEnergies[m]-dftfeOrbitalEnergiesFirstSCF[m]-tree.gaugeShift))
# #                 print('Orbital %i error: %1.3e' %(m, tree.orbitalEnergies[m]-dftfeOrbitalEnergies[m]-tree.gaugeShift))
# 
#             print('Orthogonalizing based on new order.')
#             
#             tree.orthonormalizeOrbitals()
#             print('Compute new energies from the orthogonalized orbitals.  Hope no sorting needed.')
# #             print('Not orthogonalizing...')
#             tree.updateOrbitalEnergies(sortByEnergy=True)
# 
#                 
# #             print('Computing energies and sorting before orthogonalization')
# #             tree.updateOrbitalEnergies(sortByEnergy=True)
# #             orthonormalizedOrbitals = modifiedGramSchrmidt(orbitals,weights)
# # #             orthonormalizedOrbitals = normalizeOrbitals(orbitals,weights)
# #             for m in range(tree.nOrbitals):
# #                 normDiff = np.sqrt( np.sum( (orthonormalizedOrbitals[:,m]-oldOrbitals[:,m])**2*weights ) )
# #                 print('Residual for orbtital %i: %1.5e' %(m,normDiff))
# #                 if normDiff > orbitalResidual:
# #                     orbitalResidual = np.copy(normDiff) 
# #                 tree.importPhiOnLeaves(orthonormalizedOrbitals[:,m], m)
# # #             tree.orthonormalizeOrbitals()
# #             print()
# #             print('Computing energies again, now from the orthogonalized orbitals.  Hopefully no sorting needed.')
# #             tree.updateOrbitalEnergies(sortByEnergy=True)
#             print('Using DFT-FE results from after first SCF (after orthogonalization) ...')
#             for m in range(tree.nOrbitals):
# #                 print('Orbital %i error: %1.3e' %(m, tree.orbitalEnergies[m]-dftfeOrbitalEnergies[m]-tree.gaugeShift))
#                 print('Orbital %i error: %1.3e' %(m, tree.orbitalEnergies[m]-dftfeOrbitalEnergiesFirstSCF[m]-tree.gaugeShift))
# #             tree.orthonormalizeOrbitals()
# 
# #             print('Before Veff update')
#             newOrbitalEnergies = np.sum(tree.orbitalEnergies)
#             bandEnergyResidual = abs(newOrbitalEnergies - oldOrbitalEnergies)
#             oldOrbitalEnergies = np.copy(newOrbitalEnergies)
#             
#             tree.computeBandEnergy()
#             greenIterationsCount += 1
# #             print('Sum of orbital energies after %i iterations in SCF #%i:  %f' %(greenIterationsCount,greenIterationCounter,newOrbitalEnergies))
# #             print()
#             print('Band energy after %i iterations in SCF #%i:  %1.6f H, %1.2e H' 
#                   %(greenIterationsCount,greenIterationCounter,tree.totalBandEnergy, tree.totalBandEnergy-Eband))
#             print('Band energy residual: ', bandEnergyResidual)
#             print()
#             print()
#             print()
# #             SCFiterationOutFile = '/home/njvaughn/CO_LW3_1200_SCF1_subtracting.csv'
# #             SCFiterationOutFile = '/home/njvaughn/CO_LW3_1200_SCF1_skipping_DFTFEenergies.csv'
# 
# #             SCFiterationOutFile = '/home/njvaughn/CO_LW3_800_SCF1_skipping_minD3.csv'
#             SCFiterationOutFile = '/home/njvaughn/CO_LW3_400_SCF1_skipping_minD4.csv'
# #             SCFiterationOutFile = '/home/njvaughn/CO_LW3_1200_SCF1_skipping.csv'
#             header = ['Iteration', 'orbitalResiduals', 'energyErrors']
#     
#             myData = [greenIterationsCount, residuals,
#                       tree.orbitalEnergies-dftfeOrbitalEnergiesFirstSCF-tree.gaugeShift]
#             
#         
#             if not os.path.isfile(SCFiterationOutFile):
#                 myFile = open(SCFiterationOutFile, 'a')
#                 with myFile:
#                     writer = csv.writer(myFile)
#                     writer.writerow(header) 
#                 
#             
#             myFile = open(SCFiterationOutFile, 'a')
#             with myFile:
#                 writer = csv.writer(myFile)
#                 writer.writerow(myData)
# 
#  
# 
#         print()
#         print()
# #         tree.orthonormalizeOrbitals()
#         tree.updateDensityAtQuadpoints()
#         tree.normalizeDensity()
#         sources = tree.extractLeavesDensity()  # extract the source point locations.  Currently, these are just all the leaf midpoints
#         targets = np.copy(sources)
#  
#         """ 
#         Compute new electron-electron potential and update pointwise potential values 
#         """
#         startCoulombConvolutionTime = timer()
#         V_coulombNew = np.zeros((len(targets)))
#         gpuPoissonConvolution[blocksPerGrid, threadsPerBlock](targets,sources,V_coulombNew)  # call the GPU convolution 
#         ###gpuPoissonConvolutionSingularitySubtract[blocksPerGrid, threadsPerBlock](targets,sources,V_coulombNew,5)  # call the GPU convolution 
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
# #         tree.updateOrbitalEnergies(correctPositiveEnergies=False, sortByEnergy=False) 
#         tree.updateTotalEnergy() 
#         print('Band energies after Veff update: %1.6f H, %1.2e H'
#               %(tree.totalBandEnergy, tree.totalBandEnergy-Eband))
#         
#         print('Using DFT-FE results from after first SCF...')
#         for m in range(tree.nOrbitals):
#             print('Orbital %i error: %1.3e' %(m, tree.orbitalEnergies[m]-dftfeOrbitalEnergiesFirstSCF[m]-tree.gaugeShift))
# #             print('Orbital %i error: %1.3e' %(m, tree.orbitalEnergies[m]-dftfeOrbitalEnergies[m]-tree.gaugeShift))
#         
# #         for m in range(tree.nOrbitals):
# #             if tree.orbitalEnergies[m] > 0:
# #                 tree.scrambleOrbital(m)
# #                 print('Scrambling orbital %i'%m)
# #         tree.updateOrbitalEnergies()
#         
# #         energyUpdateTime = timer() - startEnergyTime
# #         print('Energy Update took:                     %.4f seconds. ' %energyUpdateTime)
#         densityResidual = abs(Eold - tree.E)  # Compute the densityResidual for determining convergence
# #         densityResidual = -10 # STOP AFTER FIRST SCF
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
#         print('Total Energy:                          %.10f H, %.10e H' %(tree.E, tree.E-Etrue))
#         print('Energy Residual:                        %.3e\n\n' %densityResidual)
# 
# #         if vtkExport != False:
# #             tree.exportGreenIterationOrbital(vtkExport,greenIterationCounter)
#             
#         if vtkExport != False:
#             filename = vtkExport + '/mesh%03d'%(greenIterationCounter-1) + '.vtk'
#             tree.exportGridpoints(filename)
# 
#         printEachIteration=True
# #         SCFiterationOutFile = 'iterationConvergenceCO_LW1_1000.csv'
#         SCFiterationOutFile = 'iterationConvergenceCO_LW1_800.csv'
# #         SCFiterationOutFile = 'CO_LW3_800_SCF1.csv'
# #         SCFiterationOutFile = 'iterationConvergenceCO_800_domain15.csv'
# #         SCFiterationOutFile = 'iterationConvergenceCO_800_order5.csv'
# #         SCFiterationOutFile = 'iterationConvergenceLi_1200_domain24.csv'
# #         SCFiterationOutFile = 'iterationConvergenceLi_smoothingBoth.csv'
# #         SCFiterationOutFile = 'iterationConvergenceBe_LW3_1200_perturbed.csv'
#         if printEachIteration==True:
#             header = ['Iteration', 'orbitalEnergies', 'exchangePotential', 'correlationPotential', 
#                       'bandEnergy','exchangeEnergy', 'correlationEnergy', 'totalEnergy']
#         
#             myData = [greenIterationCounter, tree.orbitalEnergies, tree.totalVx, tree.totalVc, 
#                       tree.totalBandEnergy, tree.totalEx, tree.totalEc, tree.E]
#             
#         
#             if not os.path.isfile(SCFiterationOutFile):
#                 myFile = open(SCFiterationOutFile, 'a')
#                 with myFile:
#                     writer = csv.writer(myFile)
#                     writer.writerow(header) 
#                 
#             
#             myFile = open(SCFiterationOutFile, 'a')
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
      
# def greenIterations_KohnSham_SINGSUB(tree, intraScfTolerance, interScfTolerance, numberOfTargets, 
#                                 subtractSingularity, smoothingN, smoothingEps, auxiliaryFile='',SCFiterationOutFile='iterationConvergence.csv',
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
#     densityResidual = 1                                    # initialize the densityResidual to something that fails the convergence tolerance
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
#     while ( densityResidual > interScfTolerance ):
#         
#         print('\nSCF Count ', greenIterationCounter)
# 
#         orbitalResidual = 10
#         greenIterationsCount = 0
#         max_GreenIterationsCount = 3
#         while ( ( orbitalResidual > intraScfTolerance ) and ( greenIterationsCount < max_GreenIterationsCount) ):
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
#             greenIterationsCount += 1
# #             print('Sum of orbital energies after %i iterations in SCF #%i:  %f' %(greenIterationsCount,greenIterationCounter,newOrbitalEnergies))
#             print('Band energy after %i iterations in SCF #%i:  %1.6f H, %1.2e H' 
#                   %(greenIterationsCount,greenIterationCounter,tree.totalBandEnergy, tree.totalBandEnergy-Eband))
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
#         densityResidual = abs(Eold - tree.E)  # Compute the densityResidual for determining convergence
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
#         print('Energy Residual:                        %.3e\n\n' %densityResidual)
# 
# #         if vtkExport != False:
# #             tree.exportGreenIterationOrbital(vtkExport,greenIterationCounter)
# 
#         printEachIteration=True
# #         SCFiterationOutFile = 'iterationConvergenceLi_800.csv'
# #         SCFiterationOutFile = 'iterationConvergenceLi_1200_domain24.csv'
# #         SCFiterationOutFile = 'iterationConvergenceLi_smoothingBoth.csv'
# #         SCFiterationOutFile = 'iterationConvergenceBe_LW3_1200_perturbed.csv'
#         if printEachIteration==True:
#             header = ['Iteration', 'orbitalEnergies', 'exchangePotential', 'correlationPotential', 
#                       'bandEnergy','exchangeEnergy', 'correlationEnergy', 'totalEnergy']
#         
#             myData = [greenIterationCounter, tree.orbitalEnergies, tree.totalVx, tree.totalVc, 
#                       tree.totalBandEnergy, tree.totalEx, tree.totalEc, tree.E]
#             
#         
#             if not os.path.isfile(SCFiterationOutFile):
#                 myFile = open(SCFiterationOutFile, 'a')
#                 with myFile:
#                     writer = csv.writer(myFile)
#                     writer.writerow(header) 
#                 
#             
#             myFile = open(SCFiterationOutFile, 'a')
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
#     densityResidual = 1                                    # initialize the densityResidual to something that fails the convergence tolerance
#     Eold = -10.0
#     Etrue = trueEnergy(energyLevel)
#     
#     while ( densityResidual > interScfTolerance ):
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
#         densityResidual = abs(Eold - tree.E)  # Compute the densityResidual for determining convergence
#         print('Energy Residual:                 %.3e' %densityResidual)
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
#     :param interScfTolerance: exit condition for Green Iterations, densityResidual on the total energy
#     :param energyLevel: energy level trying to compute
#     '''
#     blocksPerGrid = (numberOfTargets + (threadsPerBlock - 1)) // threadsPerBlock  # compute the number of blocks based on N and threadsPerBlock
#     print('\nNumber of targets:   ', numberOfTargets)
#     print('Threads per block:   ', threadsPerBlock)
#     print('Blocks per grid:     ', blocksPerGrid)
# 
#     GIcounter=1                                     # initialize the counter to counter the number of iterations required for convergence
#     densityResidual = 1                                    # initialize the densityResidual to something that fails the convergence tolerance
#     Eold = -10.0
#     
#     Etrue = trueEnergy(energyLevel)
#     
#     while ( densityResidual > interScfTolerance ):
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
#         densityResidual = abs(Eold - tree.E)  # Compute the densityResidual for determining convergence
#         print('Energy Residual:                 %.3e' %densityResidual)
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