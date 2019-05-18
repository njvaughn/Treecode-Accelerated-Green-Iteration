'''
testGreenIterations.py
This is a unitTest module for testing Green iterations.  It begins by building the tree-based
adaotively refined mesh, then performs Green iterations to obtain the ground state energy
and wavefunction for the single electron hydrogen atom.  -- 03/20/2018 NV

Created on Mar 13, 2018
@author: nathanvaughn
'''
import os
import sys
import time
import inspect
import resource



if os.uname()[1] == 'Nathans-MacBook-Pro.local':
    rootDirectory = '/Users/nathanvaughn/Documents/GitHub/Greens-Functions-Iterative-Methods/3D-GreenIterations/adaptiveMesh/'
else:
    print('os.uname()[1] = ', os.uname()[1])

import unittest
import numpy as np
import pylibxc
from timeit import default_timer as timer
import itertools
import csv
from scipy.optimize import anderson
from scipy.optimize import root as scipyRoot



n=1
domainSize          = int(sys.argv[n]); n+=1
minDepth            = int(sys.argv[n]); n+=1
maxDepth            = int(sys.argv[n]); n+=1
additionalDepthAtAtoms        = int(sys.argv[n]); n+=1
order               = int(sys.argv[n]); n+=1
subtractSingularity = int(sys.argv[n]); n+=1
smoothingEps        = float(sys.argv[n]); n+=1
gaussianAlpha       = float(sys.argv[n]); n+=1
gaugeShift          = float(sys.argv[n]); n+=1
divideCriterion     = str(sys.argv[n]); n+=1
divideParameter1    = float(sys.argv[n]); n+=1
divideParameter2    = float(sys.argv[n]); n+=1
energyTolerance     = float(sys.argv[n]); n+=1
scfTolerance        = float(sys.argv[n]); n+=1
outputFile          = str(sys.argv[n]); n+=1
inputFile           = str(sys.argv[n]); n+=1
srcdir              = str(sys.argv[n]); n+=1
vtkDir              = str(sys.argv[n]); n+=1
noGradients         = str(sys.argv[n]) ; n+=1
symmetricIteration  = str(sys.argv[n]) ; n+=1
mixingScheme        = str(sys.argv[n]); n+=1
mixingParameter     = float(sys.argv[n]); n+=1
mixingHistoryCutoff = int(sys.argv[n]) ; n+=1
GPUpresent          = str(sys.argv[n]); n+=1
treecode            = str(sys.argv[n]); n+=1
treecodeOrder       = int(sys.argv[n]); n+=1
theta               = float(sys.argv[n]); n+=1
maxParNode          = int(sys.argv[n]); n+=1
batchSize           = int(sys.argv[n]); n+=1
divideParameter3    = float(sys.argv[n]); n+=1
divideParameter4    = float(sys.argv[n]); n+=1
base                = float(sys.argv[n]); n+=1
restart             = str(sys.argv[n]); n+=1
savedMesh           = str(sys.argv[n]); n+=1



divideParameter1 *= base
divideParameter2 *= base
divideParameter3 *= base
divideParameter4 *= base

# Set up paths based on srcdir
inputFile = srcdir+inputFile
print('inputFile = ', inputFile)
sys.path.append(srcdir+'dataStructures')
sys.path.append(srcdir+'Green-Iteration-Routines')
sys.path.append(srcdir+'utilities')
sys.path.append(srcdir+'../ctypesTests/src')

from TreeStruct_CC import Tree
import densityMixingSchemes as densityMixing

# depthAtAtoms += int(np.log2(base))
# print('Depth at atoms: ', depthAtAtoms)


print('gradientFree = ', noGradients)
print('Mixing scheme = ', mixingScheme)
print('vtk directory = ', vtkDir)

if savedMesh == 'None':
    savedMesh=''

if noGradients=='True':
    gradientFree=True
elif noGradients=='False':
    gradientFree=False
elif noGradients=='Laplacian':
    gradientFree='Laplacian'
else:
    print('Warning, not correct input for gradientFree')
    
if symmetricIteration=='True':
    symmetricIteration=True
elif symmetricIteration=='False':
    symmetricIteration=False
else:
    print('Warning, not correct input for gradientFree')

if restart=='True':
    restart=True
elif restart=='False':
    restart=False
else:
    print('Warning, not correct input for restart')
    
if GPUpresent=='True':
    GPUpresent=True
elif GPUpresent=='False':
    GPUpresent=False
else:
    print('Warning, not correct input for GPUpresent')
if treecode=='True':
    treecode=True
elif treecode=='False':
    treecode=False
else:
    print('Warning, not correct input for treecode')

# coordinateFile      = str(sys.argv[12])
# auxiliaryFile      = str(sys.argv[13])
# nElectrons          = int(sys.argv[14])
# nOrbitals          = int(sys.argv[15])
# outFile             = str(sys.argv[16])
# vtkFileBase         = str(sys.argv[17])
vtkFileBase='/home/njvaughn/results_CO/orbitals'

def clenshawCurtisNormClosure(W):
    def clenshawCurtisNorm(psi):
        norm = np.sqrt( np.sum( psi*psi*W ) )
        return norm
    return clenshawCurtisNorm

def setUpTree(onlyFillOne=False):
    '''
    setUp() gets called before every test below.
    '''
    xmin = ymin = zmin = -domainSize
    xmax = ymax = zmax = domainSize
    

    [coordinateFile, referenceEigenvaluesFile, DummyOutputFile] = np.genfromtxt(inputFile,dtype="|U100")[:3]
    [Eband, Ekinetic, Eexchange, Ecorrelation, Eelectrostatic, Etotal] = np.genfromtxt(inputFile)[3:]
    
    print('Reading atomic coordinates from: ', coordinateFile)
    atomData = np.genfromtxt(srcdir+coordinateFile,delimiter=',',dtype=float)
    print(atomData)
    if np.shape(atomData)==(5,):
        nElectrons = atomData[3]
    else:
        nElectrons = 0
        for i in range(len(atomData)):
            nElectrons += atomData[i,3]
    
    nOrbitals = int( np.ceil(nElectrons/2)  )   # start with the minimum number of orbitals 
#     nOrbitals = int( np.ceil(nElectrons/2) + 1 )   # start with the minimum number of orbitals plus 1.  
                                            # If the final orbital is unoccupied, this amount is enough. 
                                            # If there is a degeneracy leading to teh final orbital being 
                                            # partially filled, then it will be necessary to increase nOrbitals by 1.
                        
    # For O2, init 10 orbitals.
#     nOrbitals=10                    

    occupations = 2*np.ones(nOrbitals)
#     nOrbitals=7
#     print('Setting nOrbitals to six for purposes of testing the adaptivity on the oxygen atom.')
#     print('Setting nOrbitals to seven for purposes of running Carbon monoxide.')
    
    
#     nOrbitals = 6

    if inputFile=='utilities/molecularConfigurations/oxygenAtomAuxiliary.csv':
        nOrbitals=5
        occupations = 2*np.ones(nOrbitals)
        occupations[2] = 4/3
        occupations[3] = 4/3
        occupations[4] = 4/3
        
    elif inputFile=='utilities/molecularConfigurations/benzeneAuxiliary.csv':
        nOrbitals=22
        occupations = 2*np.ones(nOrbitals)
        occupations[-1]=0
#         occupations = [2, 2, 2/3 ,2/3 ,2/3, 
#                        2, 2, 2/3 ,2/3 ,2/3,
#                        2, 2, 2/3 ,2/3 ,2/3,
#                        2, 2, 2/3 ,2/3 ,2/3,
#                        2, 2, 2/3 ,2/3 ,2/3,
#                        2, 2, 2/3 ,2/3 ,2/3, 
#                        1,
#                        1,
#                        1,
#                        1,
#                        1,
#                        1]
        
    elif inputFile=='utilities/molecularConfigurations/O2Auxiliary.csv':
        nOrbitals=10
        occupations = [2,2,2,2,4/3,4/3,4/3,4/3,4/3,4/3]
        
    elif inputFile=='utilities/molecularConfigurations/carbonMonoxideAuxiliary.csv':
#         nOrbitals=10
#         occupations = [2, 2, 4/3 ,4/3 ,4/3, 
#                        2, 2, 2/3 ,2/3 ,2/3 ]
        nOrbitals=7
        occupations = 2*np.ones(nOrbitals)
    
    elif inputFile=='utilities/molecularConfigurations/hydrogenMoleculeAuxiliary.csv':
        nOrbitals=1
        occupations = [2]
        
    print('in testBatchGreen..., nOrbitals = ', nOrbitals) 
    
    print([coordinateFile, outputFile, nElectrons, nOrbitals, 
     Etotal, Eexchange, Ecorrelation, Eband, gaugeShift])
    
    referenceEigenvalues = np.array( np.genfromtxt(srcdir+referenceEigenvaluesFile,delimiter=',',dtype=float) )
    print(referenceEigenvalues)
    print(np.shape(referenceEigenvalues))
    tree = Tree(xmin,xmax,order,ymin,ymax,order,zmin,zmax,order,nElectrons,nOrbitals,additionalDepthAtAtoms=additionalDepthAtAtoms,minDepth=minDepth,gaugeShift=gaugeShift,
                coordinateFile=srcdir+coordinateFile,smoothingEps=smoothingEps, inputFile=srcdir+inputFile)#, iterationOutFile=outputFile)
    tree.referenceEigenvalues = np.copy(referenceEigenvalues)
    tree.occupations = occupations
   
    
    print('max depth ', maxDepth)
    tree.buildTree( maxLevels=maxDepth, initializationType='atomic',divideCriterion=divideCriterion, 
                    divideParameter1=divideParameter1, divideParameter2=divideParameter2, divideParameter3=divideParameter3, divideParameter4=divideParameter4, 
                    savedMesh=savedMesh, restart=restart, printTreeProperties=True,onlyFillOne=onlyFillOne)

 
    
    X,Y,Z,W,RHO,orbitals = tree.extractXYZ()
    (nPoints, nOrbitals) = np.shape(orbitals)
    print('nPoints: ', nPoints)
    print('nOrbitals: ', nOrbitals)
    atoms = tree.atoms
    return X,Y,Z,W,RHO,orbitals,atoms,nPoints,nOrbitals,nElectrons,referenceEigenvalues
     
    

def testGreenIterationsGPU_rootfinding(X,Y,Z,W,RHO,orbitals,atoms,nPoints,nOrbitals,nElectrons,referenceEigenvalues,vtkExport=False,onTheFlyRefinement=False, maxOrbitals=None, maxSCFIterations=None, restartFile=None):
    
    startTime = time.time()
    

    
    Energies, Times = greenIterations_KohnSham_SCF_rootfinding(X,Y,Z,W,RHO,orbitals,atoms,nPoints,nOrbitals,nElectrons,referenceEigenvalues,scfTolerance, energyTolerance, gradientFree, symmetricIteration, GPUpresent, treecode, treecodeOrder, theta, maxParNode, batchSize, 
                                 mixingScheme, mixingParameter, mixingHistoryCutoff,
                                 subtractSingularity, gaussianAlpha, gaugeShift,
                                 inputFile=inputFile,outputFile=outputFile, restartFile=restart,
                                 onTheFlyRefinement=onTheFlyRefinement, vtkExport=False, maxOrbitals=maxOrbitals, maxSCFIterations=maxSCFIterations)

#     greenIterations_KohnSham_SINGSUB(tree, scfTolerance, energyTolerance, nPoints, subtractSingularity, 
#                                 smoothingEps, gaussianAlpha,auxiliaryFile=auxiliaryFile, 
#                                 onTheFlyRefinement=onTheFlyRefinement, vtkExport=vtkExport)

    Times['totalKohnShamTime'] = time.time()-startTime
    print('Total Time: ', Times['totalKohnShamTime'])

    header = ['domainSize','minDepth','maxDepth','additionalDepthAtAtoms','depthAtAtoms','order','numberOfCells','numberOfPoints','gradientFree',
              'divideCriterion','divideParameter1','divideParameter2','divideParameter3','divideParameter4',
              'gaussianAlpha','gaugeShift','VextSmoothingEpsilon','energyTolerance',
              'GreenSingSubtracted', 'orbitalEnergies', 'BandEnergy', 'KineticEnergy',
              'ExchangeEnergy','CorrelationEnergy','HartreeEnergy','TotalEnergy',
              'Treecode','treecodeOrder','theta','maxParNode','batchSize','totalTime','timePerConvolution','totalIterationCount']
    
    myData = [domainSize,0,0,0,0,order,nPoints/order**3,nPoints,gradientFree,
              divideCriterion,divideParameter1,divideParameter2,divideParameter3,divideParameter4,
              gaussianAlpha,gaugeShift,smoothingEps,energyTolerance,
              subtractSingularity,
              Energies['orbitalEnergies']-Energies['gaugeShift'], Energies['Eband'], Energies['kinetic'], Energies['Ex'], Energies['Ec'], Energies['Ehartree'], Energies['Etotal'],
              treecode,treecodeOrder,theta,maxParNode,batchSize, Times['totalKohnShamTime'],Times['timePerConvolution'],Times['totalIterationCount'] ]
#               Energies['Etotal'], tree.
#               Energies['Etotal'], Energies['orbitalEnergies'][0], abs(Energies['Etotal']+1.1373748), abs(Energies['orbitalEnergies'][0]+0.378665)]
    

    runComparisonFile = os.path.split(outputFile)[0] + '/runComparison.csv'
    
    if not os.path.isfile(runComparisonFile):
        myFile = open(runComparisonFile, 'a')
        with myFile:
            writer = csv.writer(myFile)
            writer.writerow(header) 
        
        
    
    myFile = open(runComparisonFile, 'a')
    with myFile:
        writer = csv.writer(myFile)
        writer.writerow(myData)    
    


    

 

    
     
# import treecodeWrappers





xi=yi=zi=-1.1
xf=yf=zf=1.1
numpts=3000

from scfFixedPoint import scfFixedPointClosure

def greenIterations_KohnSham_SCF_rootfinding(X,Y,Z,W,RHO,orbitals,atoms,nPoints,nOrbitals,nElectrons,referenceEigenvalues,intraScfTolerance, interScfTolerance, gradientFree, symmetricIteration, GPUpresent, 
                                 treecode, treecodeOrder, theta, maxParNode, batchSize,
                                 mixingScheme, mixingParameter, mixingHistoryCutoff,
                                subtractSingularity, gaussianAlpha, gaugeShift, inputFile='',outputFile='',restartFile=False,
                                onTheFlyRefinement = False, vtkExport=False, outputErrors=False, maxOrbitals=None, maxSCFIterations=None): 
    '''
    Green Iterations for Kohn-Sham DFT using Clenshaw-Curtis quadrature.
    '''

    polarization="unpolarized"
    exchangeFunctional="LDA_X"
    correlationFunctional="LDA_C_PZ"
    exchangeFunctional = pylibxc.LibXCFunctional(exchangeFunctional, polarization)
    correlationFunctional = pylibxc.LibXCFunctional(correlationFunctional, polarization)
    
    Vext = np.zeros(nPoints)
    for atom in atoms:
        Vext += atom.V(X,Y,Z)
        
    print('Does X exist in greenIterations_KohnSham_SCF_rootfinding()? ', len(X))
    print('Does RHO exist in greenIterations_KohnSham_SCF_rootfinding()? ', len(RHO))
    
    Energies={}
    Energies['orbitalEnergies'] = np.zeros(nOrbitals)
    Energies['gaugeShift'] = gaugeShift
    Energies['kinetic'] = 0.0
    Energies['Enuclear'] = 0.0
    
    for atom1 in atoms:
        for atom2 in atoms:
            if atom1!=atom2:
                r = sqrt( (atom1.x-atom2.x)**2 + (atom1.y-atom2.y)**2 + (atom1.z-atom2.z)**2 )
                Energies['Enuclear'] += atom1.atomicNumber*atom2.atomicNumber/r
    Energies['Enuclear'] /= 2 # because of double counting
    
    Times={}
    
    
    
    
#     return
    print('MEMORY USAGE: ', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    print()

    
    
    # Store Tree variables locally
    gaugeShift = Energies['gaugeShift']
    
    

    greenIterationOutFile = outputFile[:-4]+'_GREEN_'+str(nPoints)+outputFile[-4:]
    SCFiterationOutFile =   outputFile[:-4]+'_SCF_'+str(nPoints)+outputFile[-4:]
    densityPlotsDir =       outputFile[:-4]+'_SCF_'+str(nPoints)+'_plots'
    restartFilesDir =       '/home/njvaughn/restartFiles/'+'restartFiles_'+str(nPoints)
#     restartFilesDir =       '/home/njvaughn/restartFiles/restartFiles_1416000_after25'
#     restartFilesDir =       '/Users/nathanvaughn/Documents/synchronizedDataFiles/restartFiles_1416000_after25'
    wavefunctionFile =      restartFilesDir+'/wavefunctions'
    densityFile =           restartFilesDir+'/density'
    inputDensityFile =      restartFilesDir+'/inputdensity'
    outputDensityFile =     restartFilesDir+'/outputdensity'
    vHartreeFile =          restartFilesDir+'/vHartree'
    auxiliaryFile =         restartFilesDir+'/auxiliary'
    
    plotSliceOfDensity=False
    if plotSliceOfDensity==True:
        try:
            os.mkdir(densityPlotsDir)
        except OSError:
            print('Unable to make directory ', densityPlotsDir)
        
    try:
        os.mkdir(restartFilesDir)
    except OSError:
        print('Unable to make restart directory ', restartFilesDir)
    
    
            
    if restartFile!=False:
        orbitals = np.load(wavefunctionFile+'.npy')
        oldOrbitals = np.copy(orbitals)
        for m in range(nOrbitals): 
            tree.importPhiOnLeaves(orbitals[:,m], m)
        RHO = np.load(densityFile+'.npy')
        
        inputDensities = np.load(inputDensityFile+'.npy')
        outputDensities = np.load(outputDensityFile+'.npy')
        
        V_hartreeNew = np.load(vHartreeFile+'.npy')
        
        
        # make and save dictionary
        auxiliaryRestartData = np.load(auxiliaryFile+'.npy').item()
        print('type of aux: ', type(auxiliaryRestartData))
        SCFcount = auxiliaryRestartData['SCFcount']
        Times['totalIterationCount'] = auxiliaryRestartData['totalIterationCount']
        Energies['orbitalEnergies'] = auxiliaryRestartData['eigenvalues'] 
        Eold = auxiliaryRestartData['Eold']
        
        
        
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
        
        
    
    else: 
        Eold = -10
        SCFcount=0
        Times['totalIterationCount'] = 0

        inputDensities = np.zeros((nPoints,1))
        outputDensities = np.zeros((nPoints,1))
        
        inputDensities[:,0] = np.copy(RHO)
        oldOrbitals = np.copy(orbitals)


    if plotSliceOfDensity==True:
        densitySliceSavefile = densityPlotsDir+'/densities'
        print()
        r, rho = tree.interpolateDensity(xi,yi,zi,xf,yf,zf, numpts, plot=False, save=False)
        
        densities = np.concatenate( (np.reshape(r, (numpts,1)), np.reshape(rho, (numpts,1))), axis=1)
        np.save(densitySliceSavefile,densities)

    
    
    threadsPerBlock = 512
    blocksPerGrid = (nPoints + (threadsPerBlock - 1)) // threadsPerBlock  # compute the number of blocks based on N and threadsPerBlock
    
    print('\nEntering greenIterations_KohnSham_SCF()')
    print('\nNumber of targets:   ', nPoints)
    print('Threads per block:   ', threadsPerBlock)
    print('Blocks per grid:     ', blocksPerGrid)
    
    densityResidual = 10                                   # initialize the densityResidual to something that fails the convergence tolerance

    [Eband, Ekinetic, Eexchange, Ecorrelation, Ehartree, Etotal] = np.genfromtxt(inputFile)[3:9]
    print([Eband, Ekinetic, Eexchange, Ecorrelation, Ehartree, Etotal])

 

    energyResidual=1
    residuals = 10*np.ones_like(Energies['orbitalEnergies'])
    
    referenceEnergies = {'Etotal':Etotal,'Eband':Eband,'Ehartree':Ehartree,'Eexchange':Eexchange,'Ecorrelation':Ecorrelation}
    scf_args={'inputDensities':inputDensities,'outputDensities':outputDensities,'SCFcount':SCFcount,'nPoints':nPoints,'nOrbitals':nOrbitals,'mixingHistoryCutoff':mixingHistoryCutoff,
               'GPUpresent':GPUpresent,'treecode':treecode,'treecodeOrder':treecodeOrder,'theta':theta,'maxParNode':maxParNode,'batchSize':batchSize,'alphasq':gaussianAlpha*gaussianAlpha,
               'Energies':Energies,'Times':Times,'exchangeFunctional':exchangeFunctional,'correlationFunctional':correlationFunctional,
               'Vext':Vext,'gaugeShift':gaugeShift,'orbitals':orbitals,'oldOrbitals':oldOrbitals,'subtractSingularity':subtractSingularity,
               'X':X,'Y':Y,'Z':Z,'W':W,'gradientFree':gradientFree,'residuals':residuals,'greenIterationOutFile':greenIterationOutFile,
               'threadsPerBlock':threadsPerBlock,'blocksPerGrid':blocksPerGrid,'referenceEigenvalues':referenceEigenvalues,'symmetricIteration':symmetricIteration,
               'intraScfTolerance':intraScfTolerance,'nElectrons':nElectrons,'referenceEnergies':referenceEnergies,'SCFiterationOutFile':SCFiterationOutFile,
               'wavefunctionFile':wavefunctionFile,'densityFile':densityFile,'outputDensityFile':outputDensityFile,'inputDensityFile':inputDensityFile,'vHartreeFile':vHartreeFile,
               'auxiliaryFile':auxiliaryFile}
    

    
    clenshawCurtisNorm = clenshawCurtisNormClosure(W)
    method='anderson'
    jacobianOptions={'alpha':1.0, 'M':10, 'w0':0.01} 
    solverOptions={'fatol':interScfTolerance, 'tol_norm':clenshawCurtisNorm, 'jac_options':jacobianOptions,'maxiter':1000, 'line_search':None, 'disp':True}

    
    print('Calling scipyRoot with %s method' %method)
    scfFixedPoint, scf_args = scfFixedPointClosure(scf_args)
#     print(np.shaoe(RHO))
    sol = scipyRoot(scfFixedPoint,RHO, args=scf_args, method=method, options=solverOptions)
    print(sol.success)
    print(sol.message)
    RHO = sol.x
    
#     while ( (densityResidual > interScfTolerance) or (energyResidual > interScfTolerance) ):  # terminate SCF when both energy and density are converged.
#         
#         ## CALL SCF FIXED POINT FUNCTION
# #         if SCFcount > 0:
# #             print('Exiting before first SCF (for testing initialized mesh accuracy)')
# #             return
#         scfFixedPoint, scf_args = scfFixedPointClosure(scf_args)
#         densityResidualVector = scfFixedPoint(RHO)
#         densityResidual=scf_args['densityResidual']
#         energyResidual=scf_args['energyResidual']
#         
# #         densityResidual = np.sqrt( np.sum( (outputDensities[:,SCFcount-1] - inputDensities[:,SCFcount-1])**2*weights ) )
# #         print('Density Residual from arrays ', densityResidual)
#         print('Shape of density histories: ', np.shape(outputDensities), np.shape(inputDensities))
#         
#         # Now compute new mixing with anderson scheme, then import onto tree. 
#   
#     
#         if mixingScheme == 'Simple':
#             print('Using simple mixing, from the input/output arrays')
#             simpleMixingDensity = mixingParameter*scf_args['inputDensities'][:,SCFcount-1] + (1-mixingParameter)*scf_args['outputDensities'][:,SCFcount-1]
#             integratedDensity = np.sum( simpleMixingDensity*W )
#             print('Integrated simple mixing density: ', integratedDensity)
#     #             tree.importDensityOnLeaves(simpleMixingDensity)
#             RHO = np.copy(simpleMixingDensity)
#         
#         elif mixingScheme == 'Anderson':
#             print('Using anderson mixing.')
#             andersonDensity = densityMixing.computeNewDensity(scf_args['inputDensities'], scf_args['outputDensities'], mixingParameter,W)
#             integratedDensity = np.sum( andersonDensity*W )
#             print('Integrated anderson density: ', integratedDensity)
#     #             tree.importDensityOnLeaves(andersonDensity)
#             RHO = np.copy(andersonDensity)
#         
#         elif mixingScheme == 'None':
#             RHO += densityResidualVector
#              
#         
#         
#         else:
#             print('Mixing must be set to either Simple, Anderson, or None')
#             return
#     
#                 
#         """ END WRITING INDIVIDUAL ITERATION TO FILE """
#      
#         
#         if Energies['Etotal'] > 0.0:                       # Check that the current guess for energy didn't go positive.  Reset it if it did. 
#             print('Warning, Energy is positive')
#             Energies['Etotal'] = -0.5
#             
#         
#         if SCFcount >= 150:
#             print('Setting density residual to -1 to exit after the 150th SCF')
#             densityResidual = -1
#             
# #         if SCFcount >= 1:
# #             print('Setting density residual to -1 to exit after the First SCF just to test treecode or restart')
# #             energyResidual = -1
# #             densityResidual = -1
#         
# 
# 
# 
#     
#     
#     print('\nConvergence to a tolerance of %f took %i iterations' %(interScfTolerance, SCFcount))
    return Energies, Times
    
    

    

    
if __name__ == "__main__": 
    #import sys;sys.argv = ['', 'Test.testName']

    print('='*70) 
    print('='*70) 
    print('='*70,'\n')  
    
 
    X,Y,Z,W,RHO,orbitals,atoms,nPoints,nOrbitals,nElectrons,referenceEigenvalues = setUpTree() 
    print('Does RHO exist? ', len(RHO)) 
    testGreenIterationsGPU_rootfinding(X,Y,Z,W,RHO,orbitals,atoms,nPoints,nOrbitals,nElectrons,referenceEigenvalues)
