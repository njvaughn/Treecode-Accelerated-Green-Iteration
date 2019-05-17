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



global rootDirectory
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


# from greenIterations import greenIterations_KohnSham_SCF#,greenIterations_KohnSham_SINGSUB
# from greenIterations_simultaneous import greenIterations_KohnSham_SCF_simultaneous
# from greenIterations_rootfinding import greenIterations_KohnSham_SCF_rootfinding

# from hydrogenPotential import trueWavefunction

# ThreeByThreeByThree = [element for element in itertools.product(range(3),range(3),range(3))]
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

global Temperature, KB, Sigma
Temperature = 200
KB = 1/315774.6
Sigma = Temperature*KB
def fermiObjectiveFunction(fermiEnergy):
            exponentialArg = (Energies['orbitalEnergies']-fermiEnergy)/Sigma
            temp = 1/(1+np.exp( exponentialArg ) )
            return nElectrons - 2 * np.sum(temp)

def setUpTree(onlyFillOne=False):
    '''
    setUp() gets called before every test below.
    '''
    xmin = ymin = zmin = -domainSize
    xmax = ymax = zmax = domainSize
    
    global referenceEigenvalues
#     [coordinateFile, outputFile, nElectrons, nOrbitals] = np.genfromtxt(inputFile,dtype=[(str,str,int,int,float,float,float,float,float)])[0:4]

#     [coordinateFile, outputFile, nElectrons, nOrbitals, 
#      Etotal, Eexchange, Ecorrelation, Eband, gaugeShift] = np.genfromtxt(inputFile,delimiter=',',dtype=[("|U100","|U100",int,int,float,float,float,float,float)])
    [coordinateFile, referenceEigenvaluesFile, DummyOutputFile] = np.genfromtxt(inputFile,dtype="|U100")[:3]
#     [nElectrons, nOrbitals, Eband, Ekinetic, Eexchange, Ecorrelation, Eelectrostatic, Etotal, gaugeShift] = np.genfromtxt(inputFile)[2:]
    [Eband, Ekinetic, Eexchange, Ecorrelation, Eelectrostatic, Etotal] = np.genfromtxt(inputFile)[3:]
#     nElectrons = int(nElectrons)
#     nOrbitals = int(nOrbitals)
    
#     nOrbitals = 7  # hard code this in for Carbon Monoxide
#     print('Hard coding nOrbitals to 7')
 #     nOrbitals = 6
#     print('Hard coding nOrbitals to 6 to give oxygen one extra')
#     nOrbitals = 1
#     print('Hard coding nOrbitals to 1')

    print('Reading atomic coordinates from: ', coordinateFile)
    atomData = np.genfromtxt(srcdir+coordinateFile,delimiter=',',dtype=float)
    print(atomData)
    global nElectrons
    if np.shape(atomData)==(5,):
        nElectrons = atomData[3]
    else:
        nElectrons = 0
        for i in range(len(atomData)):
            nElectrons += atomData[i,3]
    
    global nOrbitals, occupations
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
    
    global referenceEigenvalues
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

 
    
    global X,Y,Z,W,RHO,orbitals
    X,Y,Z,W,RHO,orbitals = tree.extractXYZ()
    global nPoints
    (nPoints, nOrbitals) = np.shape(orbitals)
    print('nPoints: ', nPoints)
    print('nOrbitals: ', nOrbitals)
    global atoms
    atoms = tree.atoms
    return tree
     
    
def clenshawCurtisNorm(psi):
    appendedWeights = np.append(W, 10.0)
    norm = np.sqrt( np.sum( psi*psi*appendedWeights ) )
    return norm

def testGreenIterationsGPU_rootfinding(vtkExport=False,onTheFlyRefinement=False, maxOrbitals=None, maxSCFIterations=None, restartFile=None):
    global tree
    
    startTime = time.time()
    

    
    greenIterations_KohnSham_SCF_rootfinding(scfTolerance, energyTolerance, nPoints, gradientFree, symmetricIteration, GPUpresent, treecode, treecodeOrder, theta, maxParNode, batchSize, 
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

import densityMixingSchemes as densityMixing
from fermiDiracDistribution import computeOccupations
import sys
import resource
sys.path.append('../ctypesTests')
sys.path.append('../ctypesTests/lib')

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
    
     
# import treecodeWrappers





xi=yi=zi=-1.1
xf=yf=zf=1.1
numpts=3000

def greenIterations_KohnSham_SCF_rootfinding(intraScfTolerance, interScfTolerance, nPoints, gradientFree, symmetricIteration, GPUpresent, 
                                 treecode, treecodeOrder, theta, maxParNode, batchSize,
                                 mixingScheme, mixingParameter, mixingHistoryCutoff,
                                subtractSingularity, gaussianAlpha, gaugeShift, inputFile='',outputFile='',restartFile=False,
                                onTheFlyRefinement = False, vtkExport=False, outputErrors=False, maxOrbitals=None, maxSCFIterations=None): 
    '''
    Green Iterations for Kohn-Sham DFT using Clenshaw-Curtis quadrature.
    '''
#     global tree, weights
    global threadsPerBlock, blocksPerGrid, SCFcount, m #greenIterationsCount
    global greenIterationOutFile
    global orbitals, oldOrbitals, nOrbitals
    global Veff, Vx, Vc, Vext
    
    global RHO # Needed because this function sets RHO = np.zeros(nPoints) at some point, so without 'global' it thinks it's a local variable.
    global occupations
    
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
    
    global Energies, Times
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
        global orbitals, oldOrbitals
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
#         
#         # Initialize orbital matrix
#         targets = tree.extractLeavesDensity()

#         orbitals = np.zeros((len(targets),tree.nOrbitals))
#         oldOrbitals = np.zeros((len(targets),tree.nOrbitals))
        
          
#         for m in range(nOrbitals):
#             # fill in orbitals
# #             targets = tree.extractPhi(m)
# #             weights = np.copy(targets[:,5])
#             oldOrbitals[:,m] = np.copy(targets[:,3])
#             orbitals[:,m] = np.copy(targets[:,3])
            
        # Initialize density history arrays
        inputDensities = np.zeros((nPoints,1))
        outputDensities = np.zeros((nPoints,1))
        
#         targets = tree.extractLeavesDensity() 
#         weights = targets[:,4]
        inputDensities[:,0] = np.copy(RHO)
        oldOrbitals = np.copy(orbitals)

#     targets = tree.extractLeavesDensity() 
#     weights = targets[:,4]
    
        
    
        
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

#     [Etrue, ExTrue, EcTrue, Eband] = np.genfromtxt(inputFile,dtype=[(str,str,int,int,float,float,float,float,float)])[4:8]
    [Eband, Ekinetic, Eexchange, Ecorrelation, Ehartree, Etotal] = np.genfromtxt(inputFile)[3:9]
    print([Eband, Ekinetic, Eexchange, Ecorrelation, Ehartree, Etotal])

    ### COMPUTE THE INITIAL HAMILTONIAN ###
#     density_targets = tree.extractLeavesDensity()  
#     density_sources = np.copy(density_targets)
#     sources = tree.extractDenstiySecondaryMesh()   # extract density on secondary mesh

    integratedDensity = np.sum( RHO*W )
    print('Integrated density: ', integratedDensity)

#     starthartreeConvolutionTime = timer()
#     alpha = gaussianAlpha
    alphasq=gaussianAlpha*gaussianAlpha
    
    
    if restartFile==False: # need to do initial Vhartree solve
        print('Using Gaussian singularity subtraction, alpha = ', gaussianAlpha)
        
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
#                 alpha = gaussianAlpha
                V_hartreeNew = treecodeWrappers.callTreedriver(numTargets, numSources, 
                                                               targetX, targetY, targetZ, targetValue, 
                                                               sourceX, sourceY, sourceZ, sourceValue, sourceWeight,
                                                               potentialType, gaussianAlpha, treecodeOrder, theta, maxParNode, batchSize)
                   
                if potentialType==2:
                    V_hartreeNew += targets[:,3]* (4*np.pi) / alphasq/2
    
            
    #         print('First few terms of V_hartreeNew: ', V_hartreeNew[:8])
            print('Convolution time: ', time.time()-start)
            
            
            
            
        elif GPUpresent==True:
            if treecode==False:
                V_hartreeNew = np.zeros(nPoints)
                start = time.time()
                densityInput = np.transpose( np.array([X,Y,Z,RHO,W]) )
                gpuHartreeGaussianSingularitySubract[blocksPerGrid, threadsPerBlock](densityInput,densityInput,V_hartreeNew,alphasq)
                print('Convolution time: ', time.time()-start)
    #             return
            elif treecode==True:

                start = time.time()
                potentialType=2 
                V_hartreeNew = treecodeWrappers.callTreedriver(nPoints, nPoints, 
                                                               np.copy(X), np.copy(Y), np.copy(Z), np.copy(RHO), 
                                                               np.copy(X), np.copy(Y), np.copy(Z), np.copy(RHO), np.copy(W),
                                                               potentialType, gaussianAlpha, treecodeOrder, theta, maxParNode, batchSize)
                
                
                print('Convolution time: ', time.time()-start)
                
            else:
                print('treecode True or False?')
                return
        
        
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
        
        
    
    
        print('Update orbital energies after computing the initial Veff.  Save them as the reference values for each cell')
#         tree.updateOrbitalEnergies(sortByEnergy=False, saveAsReference=True)
#         tree.computeBandEnergy()
        for m in range(nOrbitals):
            Energies['orbitalEnergies'][m] = np.sum( W* orbitals[:,m]**2 * Veff) * (2/3) # Attempt to guess initial orbital energy without computing kinetic
        Energies['Eband'] = np.sum( (Energies['orbitalEnergies']-Energies['gaugeShift']) * occupations)
        
#         tree.sortOrbitalsAndEnergies()
#         for m in range(nOrbitals):
#             # fill in orbitals
#             targets = tree.extractPhi(m)
#             weights = np.copy(targets[:,5])
#             oldOrbitals[:,m] = np.copy(targets[:,3])
#             orbitals[:,m] = np.copy(targets[:,3])
#         print('Orbital energies after initial sort: \n', Energies['orbitalEnergies'])
#         print('Kinetic:   ', tree.orbitalKinetic)
#         print('Potential: ', tree.orbitalPotential)
#         tree.updateTotalEnergy(gradientFree=False)

        Energies['Etotal'] = Energies['Eband'] - Energies['Ehartree'] + Energies['Ex'] + Energies['Ec'] - Energies['Vx'] - Energies['Vc'] + Energies['Enuclear']
        
        
        """
    
        Print results before SCF 1
        """
    
        print('Orbital Energies: ', Energies['orbitalEnergies']) 
    
        print('Orbital Energy Errors after initialization: ', Energies['orbitalEnergies']-referenceEigenvalues[:nOrbitals]-gaugeShift)
    
        print('Updated V_x:                           %.10f Hartree' %Energies['Vx'])
        print('Updated V_c:                           %.10f Hartree' %Energies['Vc'])
        
        print('Updated Band Energy:                   %.10f H, %.10e H' %(Energies['Eband'], Energies['Eband']-Eband) )
    #     print('Updated Kinetic Energy:                 %.10f H, %.10e H' %(Energies['kinetic'], Energies['kinetic']-Ekinetic) )
        print('Updated E_H:                            %.10f H, %.10e H' %(Energies['Ehartree'], Energies['Ehartree']-Ehartree) )
        print('Updated E_x:                           %.10f H, %.10e H' %(Energies['Ex'], Energies['Ex']-Eexchange) )
        print('Updated E_c:                           %.10f H, %.10e H' %(Energies['Ec'], Energies['Ec']-Ecorrelation) )
    #     print('Updated totalElectrostatic:            %.10f H, %.10e H' %(tree.totalElectrostatic, tree.totalElectrostatic-Eelectrostatic))
        print('Total Energy:                          %.10f H, %.10e H' %(Energies['Etotal'], Energies['Etotal']-Etotal))
        
        
        
        printInitialEnergies=True
    
        if printInitialEnergies==True:
            header = ['Iteration', 'densityResidual', 'orbitalEnergies','bandEnergy', 'kineticEnergy', 
                      'exchangeEnergy', 'correlationEnergy', 'hartreeEnergy', 'totalEnergy']
        
            myData = [0, 1, Energies['orbitalEnergies'], Energies['Eband'], Energies['kinetic'], 
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
    
    
        for m in range(nOrbitals):
            if Energies['orbitalEnergies'][m] > Energies['gaugeShift']:
                Energies['orbitalEnergies'][m] = Energies['gaugeShift'] - 1.0
    
        
        
    
        
    
#         if vtkExport != False:
#             filename = vtkExport + '/mesh%03d'%(SCFcount-1) + '.vtk'
#             Energies['Etotal']xportGridpoints(filename)
            
        
    #     if GPUpresent==False:
    #         print('Exiting after initialization because no GPU present.')
    #         return

  
    

    energyResidual=1
    global residuals
    residuals = 10*np.ones_like(Energies['orbitalEnergies'])
    
    while ( (densityResidual > interScfTolerance) or (energyResidual > interScfTolerance) ):  # terminate SCF when both energy and density are converged.
        SCFcount += 1
        print()
        print()
        print('\nSCF Count ', SCFcount)
        print('Orbital Energies: ', Energies['orbitalEnergies'])
#         if SCFcount > 0:
#             print('Exiting before first SCF (for testing initialized mesh accuracy)')
#             return
        
        if SCFcount>1:
            

            if (SCFcount-1)<mixingHistoryCutoff:
                inputDensities = np.concatenate( (inputDensities, np.reshape(RHO, (nPoints,1))), axis=1)
                print('Concatenated inputDensity.  Now has shape: ', np.shape(inputDensities))
            else:
                print('Beyond mixingHistoryCutoff.  Replacing column ', (SCFcount-1)%mixingHistoryCutoff)
#                                 print('Shape of oldOrbitals[:,m]: ', np.shape(oldOrbitals[:,m]))
                inputDensities[:,(SCFcount-1)%mixingHistoryCutoff] = np.copy(RHO)
        
     
        
    
            
        

        for m in range(nOrbitals): 
            print('Working on orbital %i' %m)
            print('MEMORY USAGE: ', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss )
            if m>=3:
                print('Saving restart files for after the psi0 and psi1 complete.')
                # save arrays 
                try:
                    np.save(wavefunctionFile, orbitals)
                     
                    np.save(densityFile, RHO)
                    np.save(outputDensityFile, outputDensities)
                    np.save(inputDensityFile, inputDensities)
                     
                    np.save(vHartreeFile, V_hartreeNew)
                     
                     
                     
                    # make and save dictionary
                    auxiliaryRestartData = {}
                    auxiliaryRestartData['SCFcount'] = SCFcount
                    auxiliaryRestartData['totalIterationCount'] = Times['totalIterationCount']
                    auxiliaryRestartData['eigenvalues'] = Energies['orbitalEnergies']
                    auxiliaryRestartData['Eold'] = Eold
             
                    np.save(auxiliaryFile, auxiliaryRestartData)
                except FileNotFoundError:
                    print('Failed to save restart files.')
#                         
                        
            greenIterationsCount=1
            gi_args = {'orbitals':orbitals,'oldOrbitals':oldOrbitals, 'Energies':Energies, 'Times':Times, 'Veff':Veff, 
                           'symmetricIteration':symmetricIteration,'GPUpresent':GPUpresent,'subtractSingularity':subtractSingularity,
                           'treecode':treecode, 'nPoints':nPoints, 'm':m, 'X':X,'Y':Y,'Z':Z,'W':W,'gradientFree':gradientFree,
                           'SCFcount':SCFcount,'greenIterationsCount':greenIterationsCount,'residuals':residuals,
                           'greenIterationOutFile':greenIterationOutFile, 'blocksPerGrid':blocksPerGrid,'threadsPerBlock':threadsPerBlock,
                           'referenceEigenvalues':referenceEigenvalues   } 
            
            
            resNorm=1 
            while resNorm>1e-3:
#             for njv in range(10):
#                 targets = tree.extractPhi(m)
#                 sources = tree.extractPhi(m)
#                 weights = np.copy(targets[:,5])
#                 orbitals[:,m] = np.copy(targets[:,3])
                
            
                # Orthonormalize orbital m before beginning Green's iteration
                n,M = np.shape(orbitals)
                orthWavefunction = modifiedGramSchmidt_singleOrbital(orbitals,W,m, n, M)
                orbitals[:,m] = np.copy(orthWavefunction)
#                 tree.importPhiOnLeaves(orbitals[:,m], m)
                psiIn = np.append( np.copy(orbitals[:,m]), Energies['orbitalEnergies'][m] )
#                 psiIn = 1/2*(np.copy(orbitals[:,m]) + np.copy(oldOrbitals[:,m]) )
#                 gi_args = {'orbitals':orbitals,'oldOrbitals':oldOrbitals, 'Energies':Energies, 'Times':Times, 'Veff':Veff, 
#                            'symmetricIteration':symmetricIteration,'GPUpresent':GPUpresent,'subtractSingularity':subtractSingularity,
#                            'treecode':treecode, 'nPoints':nPoints, 'm':m, 'X':X,'Y':Y,'Z':Z,'W':W,'gradientFree':gradientFree,
#                            'SCFcount':SCFcount,'greenIterationsCount':greenIterationsCount,'residuals':residuals,
#                            'greenIterationOutFile':greenIterationOutFile, 'blocksPerGrid':blocksPerGrid,'threadsPerBlock':threadsPerBlock,
#                            'referenceEigenvalues':referenceEigenvalues   } 
            
        
        
                greensIteration_FixedPoint, gi_args = greensIteration_FixedPoint_Closure(gi_args)
                print('Before: ', gi_args['greenIterationsCount'])
                r = greensIteration_FixedPoint(psiIn, gi_args)
                resNorm = clenshawCurtisNorm(r)
                print('CC norm of residual vector: ', resNorm)
#                 print(gi_args_out)
                print('After: ', gi_args['greenIterationsCount'])
                print('Dummy: ', gi_args['Dummy']) 
#                 print(gi_args_out['greenIterationsCount'])
             
             
            print('Power iteration tolerance met.  Beginning rootfinding now...') 
            tol=intraScfTolerance
#             tol=2e-7
#             if SCFcount==1: 
#                 tol = 1e-6
#             else:
#                 tol = 2e-5
#             if m>=6:  # tighten the non-degenerate deepest states for benzene.  Just an idea...
#                 tol = 2e-5
            Done = False
#             Done = True
#             print('Actually setting Done==True, and not entering fixed point problem.')
            while Done==False:
                try:
                    # Call anderson mixing on the Green's iteration fixed point function

                    # Orthonormalize orbital m before beginning Green's iteration
                    n,M = np.shape(orbitals)
                    orthWavefunction = modifiedGramSchmidt_singleOrbital(orbitals,W,m, n, M)
                    orbitals[:,m] = np.copy(orthWavefunction)
                     
                    psiIn = np.append( np.copy(orbitals[:,m]), Energies['orbitalEnergies'][m] )

                       
                    ### Anderson Options
                    method='anderson'
                    jacobianOptions={'alpha':1.0, 'M':5, 'w0':0.01} 
                    solverOptions={'fatol':tol, 'tol_norm':clenshawCurtisNorm, 'jac_options':jacobianOptions,'maxiter':1000, 'line_search':None, 'disp':True}
#                     solverOptions={'fatol':tol, 'tol_norm':eigenvalueNorm, 'jac_options':jacobianOptions,'maxiter':1000, 'line_search':None, 'disp':True}
#                     solverOptions={'fatol':tol, 'tol_norm':clenshawCurtisNorm, 'jac_options':jacobianOptions,'maxiter':1000, 'disp':True}
# #                     solverOptions={'fatol':1e-6, 'tol_norm':clenshawCurtisNorm, 'jac_options':jacobianOptions, 'disp':True}
                     
#                     ### Krylov Options
# #                     jac = Anderson()
#                     jac = BroydenFirst()
# #                     kjac = KrylovJacobian(inner_M=InverseJacobian(jac))
# #                     jacobianOptions={'method':'lgmres','inner_M':kjac, 'inner_maxiter':3, 'outer_k':2}
#                     jacobianOptions={'method':'lgmres','inner_M':InverseJacobian(jac)}
# #                     jacobianOptions={'method':'lgmres', 'inner_maxiter':3, 'outer_k':2}
#                     method='krylov'
# #                     solverOptions={'fatol':tol, 'tol_norm':clenshawCurtisNorm, 'line_search':None, 'disp':True, 'jac_options':jacobianOptions}
#                     solverOptions={'fatol':tol, 'tol_norm':clenshawCurtisNorm, 'disp':True, 'jac_options':jacobianOptions}
                     
                     
                    ### Broyden Options
#                     method='broyden1'
#                     jacobianOptions={'alpha':1.0}
# #                     solverOptions={'fatol':1e-6, 'line_search':None, 'disp':True, 'jac_options':jacobianOptions}
#                     solverOptions={'fatol':1e-6, 'tol_norm':clenshawCurtisNorm, 'jac_options':jacobianOptions, 'line_search':None, 'disp':True}

                    
                    print('Calling scipyRoot with %s method' %method)
                    sol = scipyRoot(greensIteration_FixedPoint,psiIn, args=gi_args, method=method, callback=printResidual, options=solverOptions)
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
            
        
        # sort by energy and compute new occupations
        
#         newOrder = np.argsort(Energies['orbitalEnergies'])
#         oldEnergies = np.copy(Energies['orbitalEnergies'])
#         for m in range(nOrbitals):
#             Energies['orbitalEnergies'][m] = oldEnergies[newOrder[m]]
            
#         tree.sortOrbitalsAndEnergies()
#         tree.computeOccupations()
#         for mm in range(nOrbitals):
#             # fill in orbitals  
#             targets = tree.extractPhi(mm)
#             weights = np.copy(targets[:,5])
#             oldOrbitals[:,mm] = np.copy(targets[:,3])
#             orbitals[:,mm] = np.copy(targets[:,3])  

            
        ## Compute occupations
        
        
        
        
        eF = brentq(fermiObjectiveFunction, Energies['orbitalEnergies'][0], 1, xtol=1e-14)
        print('Fermi energy: ', eF)
        exponentialArg = (Energies['orbitalEnergies']-eF)/Sigma
        occupations = 2*1/(1+np.exp( exponentialArg ) )  # these are # of electrons, not fractional occupancy.  Hence the 2*

#         occupations = computeOccupations(Energies['orbitalEnergies'], nElectrons, Temperature)
        print('Occupations: ', occupations)
        Energies['Eband'] = np.sum( (Energies['orbitalEnergies']-Energies['gaugeShift']) * occupations)


        print()  
        print()


        
        if maxOrbitals==1:
            print('Not updating density or anything since only computing one of the orbitals, not all.')
            return
        

#         oldDensity = tree.extractLeavesDensity()
        oldDensity = np.copy(RHO)
        
        RHO = np.zeros(nPoints)
        for m in range(nOrbitals):
            RHO += orbitals[:,m]**2 * occupations[m]
        newDensity = np.copy(RHO)
        
#         tree.updateDensityAtQuadpoints()
         
#         sources = tree.extractLeavesDensity()  # extract the source point locations.  Currently, these are just all the leaf midpoints
#         targets = np.copy(sources)
#         newDensity = np.copy(sources[:,3])
        
        if SCFcount==1: # not okay anymore because output density gets reset when tolerances get reset.
            outputDensities[:,0] = np.copy(newDensity)
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
        
#         densityResidual = np.sqrt( np.sum( (outputDensities[:,SCFcount-1] - inputDensities[:,SCFcount-1])**2*weights ) )
#         print('Density Residual from arrays ', densityResidual)
        print('Shape of density histories: ', np.shape(outputDensities), np.shape(inputDensities))
        
        # Now compute new mixing with anderson scheme, then import onto tree. 
      
        
        if mixingScheme == 'Simple':
            print('Using simple mixing, from the input/output arrays')
            simpleMixingDensity = mixingParameter*inputDensities[:,SCFcount-1] + (1-mixingParameter)*outputDensities[:,SCFcount-1]
            integratedDensity = np.sum( simpleMixingDensity*W )
            print('Integrated simple mixing density: ', integratedDensity)
#             tree.importDensityOnLeaves(simpleMixingDensity)
            RHO = np.copy(simpleMixingDensity)
        
        elif mixingScheme == 'Anderson':
            print('Using anderson mixing.')
            andersonDensity = densityMixing.computeNewDensity(inputDensities, outputDensities, mixingParameter,W)
            integratedDensity = np.sum( andersonDensity*W )
            print('Integrated anderson density: ', integratedDensity)
#             tree.importDensityOnLeaves(andersonDensity)
            RHO = np.copy(andersonDensity)
        
        elif mixingScheme == 'None':
            pass # don't touch the density
        
        
        else:
            print('Mixing must be set to either Simple, Anderson, or None')
            return
            

 
        """ 
        Compute new electron-electron potential and update pointwise potential values 
        """
#         starthartreeConvolutionTime = timer()

        
        if GPUpresent==True:
            if treecode==False:
                V_hartreeNew = np.zeros(nPoints)
                densityInput = np.transpose( np.array([X,Y,Z,RHO,W]) )
                gpuHartreeGaussianSingularitySubract[blocksPerGrid, threadsPerBlock](densityInput,densityInput,V_hartreeNew,alphasq)
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
        
        Energies['Eband'] = np.sum( (Energies['orbitalEnergies']-Energies['gaugeShift']) * occupations)
        Energies['Etotal'] = Energies['Eband'] - Energies['Ehartree'] + Energies['Ex'] + Energies['Ec'] - Energies['Vx'] - Energies['Vc'] + Energies['Enuclear']
        

        for m in range(nOrbitals):
            print('Orbital %i error: %1.3e' %(m, Energies['orbitalEnergies'][m]-referenceEigenvalues[m]-Energies['gaugeShift']))
        
        
        energyResidual = abs( Energies['Etotal'] - Eold )  # Compute the energyResidual for determining convergence
        Eold = np.copy(Energies['Etotal'])
        
        
        
        """
        Print results from current iteration
        """

        print('Orbital Energies: ', Energies['orbitalEnergies']) 

        print('Updated V_x:                           %.10f Hartree' %Energies['Vx'])
        print('Updated V_c:                           %.10f Hartree' %Energies['Vc'])
        
        print('Updated Band Energy:                   %.10f H, %.10e H' %(Energies['Eband'], Energies['Eband']-Eband) )
#         print('Updated Kinetic Energy:                 %.10f H, %.10e H' %(Energies['kinetic'], Energies['kinetic']-Ekinetic) )
        print('Updated E_Hartree:                      %.10f H, %.10e H' %(Energies['Ehartree'], Energies['Ehartree']-Ehartree) )
        print('Updated E_x:                           %.10f H, %.10e H' %(Energies['Ex'], Energies['Ex']-Eexchange) )
        print('Updated E_c:                           %.10f H, %.10e H' %(Energies['Ec'], Energies['Ec']-Ecorrelation) )
#         print('Updated totalElectrostatic:            %.10f H, %.10e H' %(tree.totalElectrostatic, tree.totalElectrostatic-Eelectrostatic))
        print('Total Energy:                          %.10f H, %.10e H' %(Energies['Etotal'], Energies['Etotal']-Etotal))
        print('Energy Residual:                        %.3e' %energyResidual)
        print('Density Residual:                       %.3e\n\n'%densityResidual)



            
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
     
        
        if Energies['Etotal'] > 0.0:                       # Check that the current guess for energy didn't go positive.  Reset it if it did. 
            print('Warning, Energy is positive')
            Energies['Etotal'] = -0.5
            
        
        if SCFcount >= 150:
            print('Setting density residual to -1 to exit after the 150th SCF')
            densityResidual = -1
            
        if SCFcount >= 1:
            print('Setting density residual to -1 to exit after the First SCF just to test treecode or restart')
            energyResidual = -1
            densityResidual = -1
        


        
    print('\nConvergence to a tolerance of %f took %i iterations' %(interScfTolerance, SCFcount))
    
    
 
def printResidual(x,f):
    r = clenshawCurtisNorm(f)
#     r = np.sqrt( np.sum(f*f*weights) )
    print('L2 Norm of Residual: ', r)
    
def updateTree(x,f):
    global tree, orbitals, oldOrbitals
    
    tree.importPhiOnLeaves(x,m)
    orbitals[:,m] = x.copy()
    oldOrbitals[:,m] = x.copy()
    r = clenshawCurtisNorm(f)
    print('L2 Norm of Residual: ', r)
    
    
if __name__ == "__main__": 
    #import sys;sys.argv = ['', 'Test.testName']

    print('='*70) 
    print('='*70) 
    print('='*70,'\n')  
    
 
    global tree 
    tree = setUpTree() 
    print('Does RHO exist? ', len(RHO)) 
    tree=None
    testGreenIterationsGPU_rootfinding()
