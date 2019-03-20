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
import resource

# from docutils.nodes import reference
sys.path.append('../src/dataStructures')
sys.path.append('../src/utilities')
sys.path.append('../ctypesTests/src')
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
from greenIterations import greenIterations_KohnSham_SCF#,greenIterations_KohnSham_SINGSUB

# from hydrogenPotential import trueWavefunction

# ThreeByThreeByThree = [element for element in itertools.product(range(3),range(3),range(3))]
n=1
domainSize          = 20
minDepth            = 3
maxDepth            = 20
additionalDepthAtAtoms        = 0
order               = 4
subtractSingularity = 1
smoothingEps        = 0.0
gaussianAlpha       = 1.0
divideCriterion     = 'LW5'
divideParameter1    = 500
divideParameter2    = 0
noGradients         = True
GPUpresent          = True
treecode            = True
treecodeOrder       = 8
theta               = 0.8
maxParNode          = 8000
batchSize           = 2000
divideParameter3    = 0
divideParameter4    = 0

inputFile='../src/utilities/molecularConfigurations/berylliumAuxiliary.csv'



# depthAtAtoms += int(np.log2(base))
# print('Depth at atoms: ', depthAtAtoms)



if noGradients=='True':
    gradientFree=True
elif noGradients=='False':
    gradientFree=False
else:
    print('Warning, not correct input for gradientFree')


    
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
#     [coordinateFile, outputFile, nElectrons, nOrbitals] = np.genfromtxt(inputFile,dtype=[(str,str,int,int,float,float,float,float,float)])[0:4]
#     [coordinateFile, outputFile, nElectrons, nOrbitals, 
#      Etotal, Eexchange, Ecorrelation, Eband, gaugeShift] = np.genfromtxt(inputFile,delimiter=',',dtype=[("|U100","|U100",int,int,float,float,float,float,float)])
    [coordinateFile, referenceEigenvaluesFile, DummyOutputFile] = np.genfromtxt(inputFile,dtype="|U100")[:3]
#     [nElectrons, nOrbitals, Eband, Ekinetic, Eexchange, Ecorrelation, Eelectrostatic, Etotal, gaugeShift] = np.genfromtxt(inputFile)[2:]
    [Eband, Ekinetic, Eexchange, Ecorrelation, Eelectrostatic, Etotal, gaugeShift] = np.genfromtxt(inputFile)[3:]
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
        nOrbitals=21
        occupations = 2*np.ones(nOrbitals)
    
        
    elif inputFile=='../src/utilities/molecularConfigurations/carbonMonoxideAuxiliary.csv':
        nOrbitals=7
        occupations = 2*np.ones(nOrbitals)
#     occupations[-1] = 0
    print('in testBatchGreen..., nOrbitals = ', nOrbitals)
    
#     print([coordinateFile, outputFile, nElectrons, nOrbitals, 
#      Etotal, Eexchange, Ecorrelation, Eband, gaugeShift])
    
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
                    printTreeProperties=True,onlyFillOne=onlyFillOne)


    
    return tree
    
    
def testMemory(tree):
    
    numberOfTargets = tree.numberOfGridpoints
    
    print()
    print()    
    print('~~~~~~~MEMORY USAGE~~~~~~~~ ')
    print( 'Peak:        ', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss )
    print( 'Current:     ', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss )
    print()
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
    Temperature = 100  # set to 200 Kelvin
    
    



    
    # Initialize orbital matrix
    targets = tree.extractLeavesDensity()
    orbitals = np.zeros((len(targets),tree.nOrbitals))
    oldOrbitals = np.zeros((len(targets),tree.nOrbitals))
    
          

        
    # Initialize density history arrays
    inputDensities = np.zeros((numberOfGridpoints,1))
    outputDensities = np.zeros((numberOfGridpoints,1))
    
    targets = tree.extractLeavesDensity() 
    weights = targets[:,4]
    inputDensities[:,0] = np.copy(targets[:,3])

 
    
        

    threadsPerBlock = 512
    blocksPerGrid = (numberOfTargets + (threadsPerBlock - 1)) // threadsPerBlock  # compute the number of blocks based on N and threadsPerBlock
    


    ### COMPUTE THE INITIAL HAMILTONIAN ###
    density_targets = tree.extractLeavesDensity()  
    density_sources = np.copy(density_targets)

    alpha = gaussianAlpha
    alphasq=alpha*alpha
    
    print('GPUpresent set to ', GPUpresent)
    print('Type: ', type(GPUpresent))
    
    count=0
    while count < 100:
        count+=1
        print()
        print()    
        print('MEMORY USAGE: ')
        print( resource.getrusage(resource.RUSAGE_SELF).ru_maxrss )
        print()
        print()
        
        
        if GPUpresent==False:
            numTargets = len(density_targets)
            numSources = len(density_sources)

            copystart = time.time()
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
            copytime=time.time()-copystart
            print('Copy time before convolution: ', copytime)
            start = time.time()
            
            if treecode==False:
                V_hartreeNew = directSumWrappers.callCompiledC_directSum_PoissonSingularitySubtract(numTargets, numSources, alphasq, 
                                                                                                      targetX, targetY, targetZ, targetValue,targetWeight, 
                                                                                                      sourceX, sourceY, sourceZ, sourceValue, sourceWeight)
    
                V_hartreeNew += targets[:,3]* (4*np.pi)/ alphasq/ 2   # Correct for exp(-r*r/alphasq)  # DONT TRUST
    
            elif treecode==True:
                

    
                potentialType=0 # shoud be 2 for Hartree w/ singularity subtraction.  Set to 0, 1, or 3 just to test other kernels quickly
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

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']

    print('='*70)
    print('='*70)
    print('='*70,'\n')
    


    tree = setUpTree()  
    
#     testGreenIterationsGPU(tree,vtkExport=False,onTheFlyRefinement=False, maxOrbitals=1, maxSCFIterations=1)
#     testGreenIterationsGPU(tree,vtkExport=False)
    testMemory(tree)