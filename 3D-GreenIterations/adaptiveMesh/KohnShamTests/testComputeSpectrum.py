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
from computeSpectrum import computeSpectrum

# from hydrogenPotential import trueWavefunction

# ThreeByThreeByThree = [element for element in itertools.product(range(3),range(3),range(3))]
n=1
domainSize          = int(sys.argv[n]); n+=1
nOrbitals           = int(sys.argv[n]); n+=1
targetEnergyEigenvalue = float(sys.argv[n]); n+=1
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

    if inputFile=='../src/utilities/molecularConfigurations/oxygenAtomAuxiliary.csv':
        nOrbitals=5
        occupations = 2*np.ones(nOrbitals)
        occupations[2] = 4/3
        occupations[3] = 4/3
        occupations[4] = 4/3
        
    elif inputFile=='../src/utilities/molecularConfigurations/benzeneAuxiliary.csv':
        nOrbitals=21
        occupations = 2*np.ones(nOrbitals)
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
        nOrbitals=20
        occupations = 2*np.ones(nOrbitals)
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
     
    
def testComputeSpectrum(tree,vtkExport=vtkDir,onTheFlyRefinement=False, maxOrbitals=None, maxSCFIterations=None, restartFile=None):
    
    
    startTime = time.time()
    tree.E = -1.0 # set initial energy guess


    numberOfTargets = tree.numberOfGridpoints                # set N to be the number of gridpoints.  These will be all the targets
    computeSpectrum(tree, nOrbitals, targetEnergyEigenvalue, scfTolerance, energyTolerance, numberOfTargets, gradientFree, symmetricIteration, GPUpresent, treecode, treecodeOrder, theta, maxParNode, batchSize, 
#     greenIterations_KohnSham_SCF(tree, scfTolerance, energyTolerance, numberOfTargets, gradientFree, symmetricIteration, GPUpresent, treecode, treecodeOrder, theta, maxParNode, batchSize, 
                                 mixingScheme, mixingParameter, mixingHistoryCutoff,
                                 subtractSingularity, gaussianAlpha,
                                 inputFile=inputFile,outputFile=outputFile, restartFile=restart,
                                 onTheFlyRefinement=onTheFlyRefinement, vtkExport=vtkExport, maxOrbitals=maxOrbitals, maxSCFIterations=maxSCFIterations)

#     greenIterations_KohnSham_SINGSUB(tree, scfTolerance, energyTolerance, numberOfTargets, subtractSingularity, 
#                                 smoothingEps, gaussianAlpha,auxiliaryFile=auxiliaryFile, 
#                                 onTheFlyRefinement=onTheFlyRefinement, vtkExport=vtkExport)

    totalKohnShamTime = time.time()-startTime
    print('Total Time: ', totalKohnShamTime)

#     header = ['domainSize','minDepth','maxDepth','additionalDepthAtAtoms','depthAtAtoms','order','numberOfCells','numberOfPoints','gradientFree',
#               'divideCriterion','divideParameter1','divideParameter2','divideParameter3','divideParameter4',
#               'gaussianAlpha','gaugeShift','VextSmoothingEpsilon','energyTolerance',
#               'GreenSingSubtracted', 'orbitalEnergies', 'BandEnergy', 'KineticEnergy',
#               'ExchangeEnergy','CorrelationEnergy','HartreeEnergy','TotalEnergy',
#               'Treecode','treecodeOrder','theta','maxParNode','batchSize','totalTime','totalIterationCount']
#     
#     myData = [domainSize,tree.minDepthAchieved,tree.maxDepthAchieved,tree.additionalDepthAtAtoms,tree.maxDepthAtAtoms,tree.px,tree.numberOfCells,tree.numberOfGridpoints,gradientFree,
#               divideCriterion,divideParameter1,divideParameter2,divideParameter3,divideParameter4,
#               gaussianAlpha,gaugeShift,smoothingEps,energyTolerance,
#               subtractSingularity,
#               tree.orbitalEnergies-tree.gaugeShift, tree.totalBandEnergy, tree.totalKinetic, tree.totalEx, tree.totalEc, tree.totalEhartree, tree.E,
#               treecode,treecodeOrder,theta,maxParNode,batchSize, totalKohnShamTime,tree.totalIterationCount]
# #               tree.E, tree.
# #               tree.E, tree.orbitalEnergies[0], abs(tree.E+1.1373748), abs(tree.orbitalEnergies[0]+0.378665)]
#     
# 
#     runComparisonFile = os.path.split(outputFile)[0] + '/runComparison.csv'
#     
#     if not os.path.isfile(runComparisonFile):
#         myFile = open(runComparisonFile, 'a')
#         with myFile:
#             writer = csv.writer(myFile)
#             writer.writerow(header) 
#         
#         
#     
#     myFile = open(runComparisonFile, 'a')
#     with myFile:
#         writer = csv.writer(myFile)
#         writer.writerow(myData)    
#     


    

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
#     tree.initializeOrbitalsFromAtomicData(onlyFillOne=onlyFillOne)
    
    return tree

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']

    print('='*70)
    print('='*70)
    print('='*70,'\n')
    


    tree = setUpTree()  
    
    testComputeSpectrum(tree,vtkExport=False)
