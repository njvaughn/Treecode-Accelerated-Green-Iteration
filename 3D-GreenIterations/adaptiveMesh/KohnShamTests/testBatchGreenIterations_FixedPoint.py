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
from _cffi_backend import callback
# from docutils.nodes import reference
sys.path.append('../src/dataStructures')
sys.path.append('../src/utilities')
sys.path.append('../ctypesTests/src')


global rootDirectory
if os.uname()[1] == 'Nathans-MacBook-Pro.local':
    rootDirectory = '/Users/nathanvaughn/Documents/GitHub/Greens-Functions-Iterative-Methods/3D-GreenIterations/adaptiveMesh/'
else:
    print('os.uname()[1] = ', os.uname()[1])

import unittest
import numpy as np
from timeit import default_timer as timer
import itertools
import csv

from TreeStruct_CC import Tree
# from greenIterations import greenIterations_KohnSham_SCF#,greenIterations_KohnSham_SINGSUB
# from greenIterations_simultaneous import greenIterations_KohnSham_SCF_simultaneous
# from greenIterations_rootfinding import greenIterations_KohnSham_SCF_rootfinding

# from hydrogenPotential import trueWavefunction

# ThreeByThreeByThree = [element for element in itertools.product(range(3),range(3),range(3))]
global Norbitals

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
    atomData = np.genfromtxt(coordinateFile,delimiter=',',dtype=float)
    print(atomData)
    if np.shape(atomData)==(5,):
        nElectrons = atomData[3]
    else:
        nElectrons = 0
        for i in range(len(atomData)):
            nElectrons += atomData[i,3]
    global nOrbitals
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

    if inputFile=='../src/utilities/molecularConfigurations/oxygenAtomAuxiliary.csv':
        nOrbitals=5
        occupations = 2*np.ones(nOrbitals)
        occupations[2] = 4/3
        occupations[3] = 4/3
        occupations[4] = 4/3
        
    elif inputFile=='../src/utilities/molecularConfigurations/benzeneAuxiliary.csv':
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
        
    elif inputFile=='../src/utilities/molecularConfigurations/O2Auxiliary.csv':
        nOrbitals=10
        occupations = [2,2,2,2,4/3,4/3,4/3,4/3,4/3,4/3]
        
    elif inputFile=='../src/utilities/molecularConfigurations/carbonMonoxideAuxiliary.csv':
#         nOrbitals=10
#         occupations = [2, 2, 4/3 ,4/3 ,4/3, 
#                        2, 2, 2/3 ,2/3 ,2/3 ]
        nOrbitals=7
        occupations = 2*np.ones(nOrbitals)
    
    elif inputFile=='../src/utilities/molecularConfigurations/hydrogenMoleculeAuxiliary.csv':
        nOrbitals=1
        occupations = [2]
        
    print('in testBatchGreen..., nOrbitals = ', nOrbitals)
    
    print([coordinateFile, outputFile, nElectrons, nOrbitals, 
     Etotal, Eexchange, Ecorrelation, Eband, gaugeShift])
    
    referenceEigenvalues = np.array( np.genfromtxt(referenceEigenvaluesFile,delimiter=',',dtype=float) )
    print(referenceEigenvalues)
    print(np.shape(referenceEigenvalues))
    tree = Tree(xmin,xmax,order,ymin,ymax,order,zmin,zmax,order,nElectrons,nOrbitals,additionalDepthAtAtoms=additionalDepthAtAtoms,minDepth=minDepth,gaugeShift=gaugeShift,
                coordinateFile=coordinateFile,smoothingEps=smoothingEps, inputFile=inputFile)#, iterationOutFile=outputFile)
    tree.referenceEigenvalues = np.copy(referenceEigenvalues)
    tree.occupations = occupations
    print('On the tree, nOrbitals = ', tree.nOrbitals)
    print('type: ', type(tree.nOrbitals))
    
    print('max depth ', maxDepth)
    tree.buildTree( maxLevels=maxDepth, initializationType='atomic',divideCriterion=divideCriterion, 
                    divideParameter1=divideParameter1, divideParameter2=divideParameter2, divideParameter3=divideParameter3, divideParameter4=divideParameter4, 
                    savedMesh=savedMesh, restart=restart, printTreeProperties=True,onlyFillOne=onlyFillOne)

 
    
    return tree
     
    
def testGreenIterationsGPU_rootfinding(vtkExport=False,onTheFlyRefinement=False, maxOrbitals=None, maxSCFIterations=None, restartFile=None):
    global tree
    
    startTime = time.time()
    tree.E = -1.0 # set initial energy guess


    numberOfTargets = tree.numberOfGridpoints                # set N to be the number of gridpoints.  These will be all the targets
    
    greenIterations_KohnSham_SCF_rootfinding(scfTolerance, energyTolerance, numberOfTargets, gradientFree, symmetricIteration, GPUpresent, treecode, treecodeOrder, theta, maxParNode, batchSize, 
                                 mixingScheme, mixingParameter, mixingHistoryCutoff,
                                 subtractSingularity, gaussianAlpha,
                                 inputFile=inputFile,outputFile=outputFile, restartFile=restart,
                                 onTheFlyRefinement=onTheFlyRefinement, vtkExport=False, maxOrbitals=maxOrbitals, maxSCFIterations=maxSCFIterations)

#     greenIterations_KohnSham_SINGSUB(tree, scfTolerance, energyTolerance, numberOfTargets, subtractSingularity, 
#                                 smoothingEps, gaussianAlpha,auxiliaryFile=auxiliaryFile, 
#                                 onTheFlyRefinement=onTheFlyRefinement, vtkExport=vtkExport)

    totalKohnShamTime = time.time()-startTime
    print('Total Time: ', totalKohnShamTime)

    header = ['domainSize','minDepth','maxDepth','additionalDepthAtAtoms','depthAtAtoms','order','numberOfCells','numberOfPoints','gradientFree',
              'divideCriterion','divideParameter1','divideParameter2','divideParameter3','divideParameter4',
              'gaussianAlpha','gaugeShift','VextSmoothingEpsilon','energyTolerance',
              'GreenSingSubtracted', 'orbitalEnergies', 'BandEnergy', 'KineticEnergy',
              'ExchangeEnergy','CorrelationEnergy','HartreeEnergy','TotalEnergy',
              'Treecode','treecodeOrder','theta','maxParNode','batchSize','totalTime','timePerConvolution','totalIterationCount']
    
    myData = [domainSize,tree.minDepthAchieved,tree.maxDepthAchieved,tree.additionalDepthAtAtoms,tree.maxDepthAtAtoms,tree.px,tree.numberOfCells,tree.numberOfGridpoints,gradientFree,
              divideCriterion,divideParameter1,divideParameter2,divideParameter3,divideParameter4,
              gaussianAlpha,gaugeShift,smoothingEps,energyTolerance,
              subtractSingularity,
              tree.orbitalEnergies-tree.gaugeShift, tree.totalBandEnergy, tree.totalKinetic, tree.totalEx, tree.totalEc, tree.totalEhartree, tree.E,
              treecode,treecodeOrder,theta,maxParNode,batchSize, totalKohnShamTime,tree.timePerConvolution,tree.totalIterationCount]
#               tree.E, tree.
#               tree.E, tree.orbitalEnergies[0], abs(tree.E+1.1373748), abs(tree.orbitalEnergies[0]+0.378665)]
    

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
    


     

def updateTree(tree, onlyFillOne=False):


#     tree.nOrbitals = nOrbitals
    tree.numberOfGridpoints = 0
    tree.numberOfCells = 0
    closestToOrigin = 10
    
    
    for _,cell in tree.masterList:
        if cell.leaf==True:
            tree.numberOfCells += 1
            for i,j,k in tree.PxByPyByPz:
                tree.numberOfGridpoints += 1
#                 cell.gridpoints[i,j,k].phi = np.zeros(nOrbitals)
                cell.gridpoints[i,j,k].counted = True
                gp = cell.gridpoints[i,j,k]
                r = np.sqrt( gp.x*gp.x + gp.y*gp.y + gp.z*gp.z )
                if r < closestToOrigin:
                    closestToOrigin = np.copy(r)
                    closestCoords = [gp.x, gp.y, gp.z]
                    closestMidpoint = [cell.xmid, cell.ymid, cell.zmid]

    tree.rmin = closestToOrigin
    
    
                    
    for _,cell in tree.masterList:
        for i,j,k in tree.PxByPyByPz:
            if hasattr(cell.gridpoints[i,j,k], "counted"):
                cell.gridpoints[i,j,k].counted = None
     
    
    print('Number of gridpoints: ', tree.numberOfGridpoints)
    tree.computeDerivativeMatrices()
    tree.initializeDensityFromAtomicData()
    tree.initializeOrbitalsFromAtomicData(onlyFillOne=onlyFillOne)
    
    return tree


import numpy as np
import os
import csv
from numba import cuda, jit, njit
import time
# from scipy.optimize import anderson as scipyAnderson
from scipy.optimize import root as scipyRoot
from scipy.optimize.nonlin import BroydenFirst, KrylovJacobian
from scipy.optimize.nonlin import InverseJacobian
# from scipy.optimize import newton_krylov as scipyNewtonKrylov

import densityMixingSchemes as densityMixing
from fermiDiracDistribution import computeOccupations
import sys
import resource
sys.path.append('../ctypesTests')
sys.path.append('../ctypesTests/lib')

# from fixedPointFunctions import greensIteration_FixedPoint

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

def clenshawCurtisNorm(psi):
#     return np.max(np.abs(psi))
#     print('USING CLENSHAW CURTIS NORM CALLED BY ', inspect.stack()[2][3])
# #     return np.sqrt(np.sum(psi*psi))
# #     global weights
#      
    appendedWeights = np.append(weights, 10.0)
#     print(appendedWeights[-5:])
    norm = np.sqrt( np.sum( psi*psi*appendedWeights ) )
#     print('Norm = ', norm)
    return norm

def eigenvalueNorm(psi):
    norm = np.sqrt( psi[-1]**2 )
    return norm


def clenshawCurtisNorm_withoutEigenvalue(psi):
    return np.sqrt( np.sum( psi*psi*weights ) )



def greensIteration_FixedPoint(psiIn):
    print('Who called F(x)? ', inspect.stack()[2][3])
    inputWave = np.copy(psiIn[:-1])
#     print('Norm of psiIn:', clenshawCurtisNorm_withoutEigenvalue(inputWave))
#     print('Norm of psiIn - what was already in orbitals array: ', clenshawCurtisNorm_withoutEigenvalue(inputWave-orbitals[:,m]))
#     print('Norm of psiIn - what was already in oldOrbitals array: ', clenshawCurtisNorm_withoutEigenvalue(inputWave-oldOrbitals[:,m]))
    
#     psiIn /= clenshawCurtisNorm(psiIn)
#     print('Normalizing psiIn...')
#     print('Norm of psiIn - what was already in orbitals array: ', clenshawCurtisNorm(psiIn-orbitals[:,m]))
#     print('Norm of psiIn - what was already in oldOrbitals array: ', clenshawCurtisNorm(psiIn-oldOrbitals[:,m]))
    # global data structures
    global tree, orbitals, oldOrbitals, residuals, eigenvalueHistory
    
    # Global constants and counters
    global threadsPerBlock, blocksPerGrid, SCFcount, greenIterationsCount
    global greenIterationOutFile
    
#     if inspect.stack()[2][3]=='nonlin_solve':
    if True:
        tree.totalIterationCount += 1
        
        
        oldOrbitals[:,m] = np.copy(psiIn[:-1])    
        orbitals[:,m] = np.copy(psiIn[:-1])
        n,M = np.shape(orbitals)
        orthWavefunction = modifiedGramSchmidt_singleOrbital(orbitals,weights,m, n, M)
        orbitals[:,m] = np.copy(orthWavefunction)
        tree.importPhiOnLeaves(orbitals[:,m], m)
        tree.orbitalEnergies[m] = np.copy(psiIn[-1])
    else:
#         print('Different function called F(x), not updating tree.')
        print('Not updating tree.')

#     tree.importPhiOnLeaves(oldOrbitals[:,m], m)
# #     oldOrbitals[:,m] = np.copy(orbitals[:,m])
    
#     targets = tree.extractPhi(m)
#     sources = np.copy(targets)


    if symmetricIteration==False:
        sources = tree.extractGreenIterationIntegrand(m)
#         sources = tree.extractGreenIterationIntegrand_Deflated(m,orbitals,weights)
    elif symmetricIteration == True:
#                     sources = tree.extractGreenIterationIntegrand_Deflated(m,orbitals,weights)
        sources, sqrtV = tree.extractGreenIterationIntegrand_symmetric(m)
    else: 
        print("symmetricIteration variable not True or False.  What should it be?")
        return
    
    
    targets=np.copy(sources)



    oldEigenvalue =  tree.orbitalEnergies[m] 
    k = np.sqrt(-2*tree.orbitalEnergies[m])

    phiNew = np.zeros((len(targets)))
    if subtractSingularity==0: 
        print('Using singularity skipping')
        gpuHelmholtzConvolution[blocksPerGrid, threadsPerBlock](targets,sources,phiNew,k) 
    elif subtractSingularity==1:
        if tree.orbitalEnergies[m] < 10.25**100: 
            
            
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


                phiNew /= (4*np.pi)
            elif GPUpresent==True:
                if treecode==False:
                    startTime = time.time()
                    if symmetricIteration==False:
                        gpuHelmholtzConvolutionSubractSingularity[blocksPerGrid, threadsPerBlock](targets,sources,phiNew,k) 
                        convolutionTime = time.time()-startTime
                        print('Using asymmetric singularity subtraction.  Convolution time: ', convolutionTime)
                    elif symmetricIteration==True:
                        gpuHelmholtzConvolutionSubractSingularitySymmetric[blocksPerGrid, threadsPerBlock](targets,sources,sqrtV,phiNew,k) 
                        phiNew *= -1
                        convolutionTime = time.time()-startTime
                        print('Using symmetric singularity subtraction.  Convolution time: ', convolutionTime)
                    convTime=time.time()-startTime
                    print('Convolution time: ', convTime)
                    tree.timePerConvolution = convTime
                    
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
                    
                    print(np.shape(targetX))
                
                    copytime=time.time()-copyStart
#                                         print('Time spent copying arrays for treecode call: ', copytime)
                    
                    potentialType=3
                    kappa = k
                    startTime = time.time()
                    numDevices=4
                    numThreads=4
                    phiNew = treecodeWrappers.callTreedriver(numTargets, numSources, 
                                                                   targetX, targetY, targetZ, targetValue, 
                                                                   sourceX, sourceY, sourceZ, sourceValue, sourceWeight,
                                                                   potentialType, kappa, treecodeOrder, theta, maxParNode, batchSize, numDevices, numThreads)
                    convTime=time.time()-startTime
                    print('Convolution time: ', convTime)
                    tree.timePerConvolution = convTime
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
    

#     if ( (gradientFree==True) and (SCFcount>-1) and False):                 
    if ( (gradientFree==True) and (SCFcount>-1)):                 
        
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
            
            # Import normalized psi*sqrtV into phiOld
            phiNew /= np.sqrt( np.sum(phiNew*phiNew*weights ))
            tree.setPhiOldOnLeaves_symmetric(phiNew)
            
            
            orbitals[:,m] = np.copy(phiNew/sqrtV)
        
        n,M = np.shape(orbitals)
#         print('Not orthgonoalizing, relying on deflation instead... (640)')
        orthWavefunction = modifiedGramSchmidt_singleOrbital(orbitals,weights,m, n, M)
        orbitals[:,m] = np.copy(orthWavefunction)
        tree.importPhiOnLeaves(orbitals[:,m], m)
        


        if greenIterationsCount==1:
            eigenvalueHistory = np.array(tree.orbitalEnergies[m])
        else:
            
            eigenvalueHistory = np.append(eigenvalueHistory, tree.orbitalEnergies[m])
        print('eigenvalueHistory: \n',eigenvalueHistory)
        
        
        print('Orbital energy after Harrison update: ', tree.orbitalEnergies[m])
         

#     elif ( (gradientFree==False) or (SCFcount==-1) and False ):
    elif ( (gradientFree==False) or (gradientFree=='Laplacian') ):
        
        # update the orbital
        if symmetricIteration==False:
            orbitals[:,m] = np.copy(phiNew)
        elif symmetricIteration==True:
            orbitals[:,m] = np.copy(phiNew/sqrtV)
        else:
            print('What should symmetricIteration equal?')
            return
          
        # Compute energy before orthogonalization, just to see  
#         tree.importPhiOnLeaves(orbitals[:,m], m)
#         tree.updateOrbitalEnergies(laplacian=gradientFree,sortByEnergy=False, targetEnergy=m)
        
        n,M = np.shape(orbitals)
        orthWavefunction = modifiedGramSchmidt_singleOrbital(orbitals,weights,m, n, M)
        orbitals[:,m] = np.copy(orthWavefunction)
        tree.importPhiOnLeaves(orbitals[:,m], m)
        
#                             tree.importPhiOnLeaves(orbitals[:,m], m)
#                             tree.orthonormalizeOrbitals(targetOrbital=m)
        
        ## Update orbital energies after orthogonalization
        tree.updateOrbitalEnergies(laplacian=gradientFree,sortByEnergy=False, targetEnergy=m) 

        
    else:
        print('Not updating eigenvalue.  Is that intended?')
#                             print('Invalid option for gradientFree, which is set to: ', gradientFree)
#                             print('type: ', type(gradientFree))

        if greenIterationsCount==1:
            eigenvalueHistory = np.array(tree.orbitalEnergies[m])
        else:
            
            eigenvalueHistory = np.append(eigenvalueHistory, tree.orbitalEnergies[m])
        print('eigenvalueHistory: \n',eigenvalueHistory)
        
        orbitals[:,m] = np.copy(phiNew)
        n,M = np.shape(orbitals)
        orthWavefunction = modifiedGramSchmidt_singleOrbital(orbitals,weights,m, n, M)
        orbitals[:,m] = np.copy(orthWavefunction)
        
        
        tree.importPhiOnLeaves(orbitals[:,m], m)
        tree.setPhiOldOnLeaves(m)
        
       
        eigenvalueHistory = np.append(eigenvalueHistory, tree.orbitalEnergies[m])
    
    
    if tree.orbitalEnergies[m]>0.0:
        tree.orbitalEnergies[m] = tree.gaugeShift - 0.5
        print('Energy eigenvalue was positive, setting to gauge shift - 0.5')
        
    tempOrbital = tree.extractPhi(m)
    orbitals[:,m] = np.copy( tempOrbital[:,3] )
    
    
#     tree.printWavefunctionNearEachAtom(m)
        
#     residualVector = orbitals[:,m] - oldOrbitals[:,m]
    psiOut = np.append(np.copy(orbitals[:,m]), np.copy(tree.orbitalEnergies[m]))
    residualVector = psiOut - psiIn
    
    
    
    loc = np.argmax(np.abs(residualVector[:-1]))
    print('Largest residual: ', residualVector[loc])
    print('Value at that point: ', psiOut[loc])
    print('Location of max residual: ', tempOrbital[loc,0], tempOrbital[loc,1], tempOrbital[loc,2])
    
    print()
    print('Max value of input wavefunction:   ', np.max(np.abs(psiIn[:-1])))
    print('Max value of output wavefunction:  ', np.max(np.abs(psiOut[:-1])))
    print()
#     residualVector = -(psiIn - orbitals[:,m]) 

    newEigenvalue = tree.orbitalEnergies[m]
    
    
        

    
    if symmetricIteration==False:
        normDiff = np.sqrt( np.sum( (orbitals[:,m]-oldOrbitals[:,m])**2*weights ) )
    elif symmetricIteration==True:
        normDiff = np.sqrt( np.sum( (orbitals[:,m]*sqrtV-oldOrbitals[:,m]*sqrtV)**2*weights ) )
    eigenvalueDiff = abs(newEigenvalue - oldEigenvalue)
    
    tree.eigenvalueDiff = eigenvalueDiff
    
    

    residuals[m] = normDiff
    orbitalResidual = np.copy(normDiff)
    
    

    print('Orbital %i error and eigenvalue residual:   %1.3e and %1.3e' %(m,tree.orbitalEnergies[m]-tree.referenceEigenvalues[m]-tree.gaugeShift, eigenvalueDiff))
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
    
      
    greenIterationsCount += 1
    return residualVector


xi=yi=zi=-1.1
xf=yf=zf=1.1
numpts=3000

def greenIterations_KohnSham_SCF_rootfinding(intraScfTolerance, interScfTolerance, numberOfTargets, gradientFree, symmetricIteration, GPUpresent, 
                                 treecode, treecodeOrder, theta, maxParNode, batchSize,
                                 mixingScheme, mixingParameter, mixingHistoryCutoff,
                                subtractSingularity, gaussianAlpha, inputFile='',outputFile='',restartFile=False,
                                onTheFlyRefinement = False, vtkExport=False, outputErrors=False, maxOrbitals=None, maxSCFIterations=None): 
    '''
    Green Iterations for Kohn-Sham DFT using Clenshaw-Curtis quadrature.
    '''
    global tree, weights
    global threadsPerBlock, blocksPerGrid, SCFcount, greenIterationsCount, m
    global greenIterationOutFile
    global orbitals, oldOrbitals
    
    
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
    
    
    
    if maxOrbitals==1:
        nOrbitals = 1
    else:
        nOrbitals = tree.nOrbitals
            
    if restartFile!=False:
        global orbitals, oldOrbitals
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
                numDevices=4
                numThreads=4
                V_hartreeNew = treecodeWrappers.callTreedriver(numTargets, numSources, 
                                                               targetX, targetY, targetZ, targetValue, 
                                                               sourceX, sourceY, sourceZ, sourceValue, sourceWeight,
                                                               potentialType, gaussianAlpha, treecodeOrder, theta, maxParNode, batchSize, numDevices, numThreads)
                   
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
                numDevices=4
                numThreads=4
                V_hartreeNew = treecodeWrappers.callTreedriver(numTargets, numSources, 
                                                               targetX, targetY, targetZ, targetValue, 
                                                               sourceX, sourceY, sourceZ, sourceValue, sourceWeight,
                                                               potentialType, gaussianAlpha, treecodeOrder, theta, maxParNode, batchSize, numDevices, numThreads)
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
    
        
        
    
        
    
#         if vtkExport != False:
#             filename = vtkExport + '/mesh%03d'%(SCFcount-1) + '.vtk'
#             tree.exportGridpoints(filename)
            
        
    #     if GPUpresent==False:
    #         print('Exiting after initialization because no GPU present.')
    #         return

    initialWaveData = tree.extractPhi(0)
    initialPsi0 = np.copy(initialWaveData[:,3])
    x = np.copy(initialWaveData[:,0])
    y = np.copy(initialWaveData[:,1])
    z = np.copy(initialWaveData[:,2])
    



    energyResidual=1
    global residuals
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
            print('Working on orbital %i' %m)
            print('MEMORY USAGE: ', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss )
            if m>=3:
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
#                         
                        
            greenIterationsCount=1

            resNorm=1
            while resNorm>1e-2:
#             for njv in range(10):
                targets = tree.extractPhi(m)
                sources = tree.extractPhi(m)
                weights = np.copy(targets[:,5])
                orbitals[:,m] = np.copy(targets[:,3])
                
            
                # Orthonormalize orbital m before beginning Green's iteration
                n,M = np.shape(orbitals)
                orthWavefunction = modifiedGramSchmidt_singleOrbital(orbitals,weights,m, n, M)
                orbitals[:,m] = np.copy(orthWavefunction)
                tree.importPhiOnLeaves(orbitals[:,m], m)
                psiIn = np.append( np.copy(orbitals[:,m]), tree.orbitalEnergies[m] )
#                 psiIn = 1/2*(np.copy(orbitals[:,m]) + np.copy(oldOrbitals[:,m]) )
                r = greensIteration_FixedPoint(psiIn)
                resNorm = clenshawCurtisNorm(r)
                print('CC norm of residual vector: ', resNorm)

            
            
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
                    targets = tree.extractPhi(m)
                    sources = tree.extractPhi(m)
                    weights = np.copy(targets[:,5])
                    orbitals[:,m] = np.copy(targets[:,3])
                     
                 
                    # Orthonormalize orbital m before beginning Green's iteration
                    n,M = np.shape(orbitals)
                    orthWavefunction = modifiedGramSchmidt_singleOrbital(orbitals,weights,m, n, M)
                    orbitals[:,m] = np.copy(orthWavefunction)
                    tree.importPhiOnLeaves(orbitals[:,m], m) 
                     
                    psiIn = np.append( np.copy(orbitals[:,m]), tree.orbitalEnergies[m] )
#                     print('Calling scipyAnderson')
#                     psiOut = scipyAnderson(greensIteration_FixedPoint,psiIn,maxiter=5, alpha=1, M=5, w0=0.01, f_tol=tol, verbose=True, callback=printResidual)
                      
                       
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
                    sol = scipyRoot(greensIteration_FixedPoint,psiIn, method=method, callback=printResidual, options=solverOptions)
                    print(sol.success)
                    print(sol.message)
                    psiOut = sol.x
                    Done = True
                except Exception:
                    if np.abs(tree.eigenvalueDiff) < tol/10:
                        print("Rootfinding didn't converge but eigenvalue is converged.  Exiting because this is probably due to degeneracy in the space.")
                        targets = tree.extractPhi(m)
                        psiOut = np.append(targets[:,3], tree.orbitalEnergies[m])
                        Done=True
                    else:
                        print('Not converged.  What to do?')
                        return
            orbitals[:,m] = np.copy(psiOut[:-1])
            tree.orbitalEnergies[m] = np.copy(psiOut[-1])
             
            print('Used %i iterations for wavefunction %i' %(greenIterationsCount,m))



#                 if eigenvalueDiff<tol:
#                     pass
#                 else:
#                     print('Anderson didnt converge, eigenvalue not converged, what to do??  Try again?')
#                     targets = tree.extractPhi(m)
#                     sources = tree.extractPhi(m)
#                     weights = np.copy(targets[:,5])
#                     orbitals[:,m] = np.copy(targets[:,3])
#                     
#                 
#                     # Orthonormalize orbital m before beginning Green's iteration
#                     n,M = np.shape(orbitals)
#                     orthWavefunction = modifiedGramSchmidt_singleOrbital(orbitals,weights,m, n, M) 
#                     orbitals[:,m] = np.copy(orthWavefunction)
#                     tree.importPhiOnLeaves(orbitals[:,m], m) 
#                     
#                     psiIn = np.append( np.copy(orbitals[:,m]), tree.orbitalEnergies[m] )
#                     psiOut = scipyAnderson(greensIteration_FixedPoint,psiIn,alpha=1, M=5, w0=0.01, max_iter=30, line_search=None, f_tol=tol, verbose=True, callback=printResidual)


#                 targets = tree.extractPhi(m)
#                 sources = tree.extractPhi(m)
#                 weights = np.copy(targets[:,5])
#                 orbitals[:,m] = np.copy(targets[:,3])
#                 
#             
#                 # Orthonormalize orbital m before beginning Green's iteration
#                 n,M = np.shape(orbitals)
#                 orthWavefunction = modifiedGramSchmidt_singleOrbital(orbitals,weights,m, n, M)
#                 orbitals[:,m] = np.copy(orthWavefunction)
#                 tree.importPhiOnLeaves(orbitals[:,m], m)
#                 psiIn = np.append( np.copy(orbitals[:,m]), tree.orbitalEnergies[m] )
# #                 psiIn = 1/2*(np.copy(orbitals[:,m]) + np.copy(oldOrbitals[:,m]) )
#                 r = greensIteration_FixedPoint(psiIn)
#                 resNorm = clenshawCurtisNorm(r)
#                 print('CC norm of residual vector: ', resNorm)
                
                
#             psiOut = scipyAnderson(greensIteration_FixedPoint,psiIn,alpha=1, M=10, w0=0.01,line_search=None,tol_norm=clenshawCurtisNorm, f_tol=1e-4, verbose=True, callback=printResidual)
#             psiOut = scipyAnderson(greensIteration_FixedPoint,psiIn,alpha=1, M=10, w0=0.01,line_search=None, f_tol=1e-4, verbose=True, callback=printResidual)
#             psiOut = scipyNewtonKrylov(greensIteration_FixedPoint,psiIn, inner_maxiter=4, f_tol=1e-4, tol_norm=clenshawCurtisNorm, verbose=True, callback=printResidual)            
            
            
            
        
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
        
#         print('Sample of output densities:')
#         print(outputDensities[0,:])    
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
#                 alpha = gaussianAlpha
                numThreads=4
                numDevices=4
                V_hartreeNew = treecodeWrappers.callTreedriver(numTargets, numSources, 
                                                               targetX, targetY, targetZ, targetValue, 
                                                               sourceX, sourceY, sourceZ, sourceValue, sourceWeight,
                                                               potentialType, gaussianAlpha, treecodeOrder, theta, maxParNode, batchSize, numDevices, numThreads)
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
#                 numDevices=4
#                 numThreads=4
#                 V_hartreeNew = treecodeWrappers.callTreedriver(numTargets, numSources, 
#                                                                targetX, targetY, targetZ, targetValue, 
#                                                                sourceX, sourceY, sourceZ, sourceValue, sourceWeight,
#                                                                potentialType, alphasq, order, theta, maxParNode, batchSize, numDevices, numThreads)
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



            
#         if vtkExport != False:
#             filename = vtkExport + '/mesh%03d'%(SCFcount-1) + '.vtk'
#             tree.exportGridpoints(filename)

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
            
#         if SCFcount >= 1:
#             print('Setting density residual to -1 to exit after the First SCF just to test treecode or restart')
#             energyResidual = -1
#             densityResidual = -1
        


        
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
    
#     testGreenIterationsGPU(tree,vtkExport=False,onTheFlyRefinement=False, maxOrbitals=1, maxSCFIterations=1)
    testGreenIterationsGPU_rootfinding()
