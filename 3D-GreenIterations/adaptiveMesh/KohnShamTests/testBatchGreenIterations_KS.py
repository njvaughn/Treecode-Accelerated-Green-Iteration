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
sys.path.append('../src/dataStructures')
sys.path.append('../src/utilities')

import unittest
import numpy as np
from timeit import default_timer as timer
import itertools
import csv

from TreeStruct_CC import Tree
from greenIterations import greenIterations_KohnSham_SCF#,greenIterations_KohnSham_SINGSUB

# from hydrogenPotential import trueWavefunction

# ThreeByThreeByThree = [element for element in itertools.product(range(3),range(3),range(3))]

domainSize          = int(sys.argv[1])
minDepth            = int(sys.argv[2])
maxDepth            = int(sys.argv[3])
order               = int(sys.argv[4])
subtractSingularity = int(sys.argv[5])
smoothingN          = int(sys.argv[6])
smoothingEps        = float(sys.argv[7])
divideCriterion     = str(sys.argv[8])
divideParameter     = float(sys.argv[9])
energyTolerance     = float(sys.argv[10])
scfTolerance        = float(sys.argv[11])
outputFile          = str(sys.argv[12])
inputFile           = str(sys.argv[13])
noGradients         = str(sys.argv[14])
mixingScheme        = str(sys.argv[15])
mixingParameter     = float(sys.argv[16])

print('gradientFree = ', noGradients)
print('Mixing scheme = ', mixingScheme)

if noGradients=='True':
    gradientFree=True
elif noGradients=='False':
    gradientFree=False
else:
    print('Warning, not correct input for gradientFree')

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
    [coordinateFile, DummyOutputFile] = np.genfromtxt(inputFile,dtype="|U100")[:2]
    [nElectrons, nOrbitals, Eband, Ekinetic, Eexchange, Ecorrelation, Eelectrostatic, Etotal, gaugeShift] = np.genfromtxt(inputFile)[2:]
    nElectrons = int(nElectrons)
    nOrbitals = int(nOrbitals)
    
    nOrbitals = 7  # hard code this in for Carbon Monoxide
    print('Hard coding nOrbitals to 7')

#     nOrbitals = 6
#     print('Hard coding nOrbitals to 6 to give oxygen one extra')
#     nOrbitals = 1
#     print('Hard coding nOrbitals to 1')
    
    
    print([coordinateFile, outputFile, nElectrons, nOrbitals, 
     Etotal, Eexchange, Ecorrelation, Eband, gaugeShift])
    tree = Tree(xmin,xmax,order,ymin,ymax,order,zmin,zmax,order,nElectrons,nOrbitals,maxDepthAtAtoms=maxDepth,gaugeShift=gaugeShift,
                coordinateFile=coordinateFile,inputFile=inputFile)#, iterationOutFile=outputFile)

    
    
    print('max depth ', maxDepth)
    tree.buildTree( minLevels=minDepth, maxLevels=maxDepth, initializationType='atomic',divideCriterion=divideCriterion, divideParameter=divideParameter, printTreeProperties=True,onlyFillOne=onlyFillOne)
#     for element in tree.masterList:
#         
# #             element[1].gridpoints[1,1,1].setPsi(np.random.rand(1))
#         for i,j,k in tree.PxByPyByPz:
#             element[1].gridpoints[i,j,k].setPsi(np.random.rand(1))
            
#     for m in range(5,tree.nOrbitals):
#         tree.scrambleOrbital(m)
#     tree.normalizeDensity()

#     tree.computeOrbitalMoments()
    

    
    return tree
    
    
def testGreenIterationsGPU(tree,vtkExport=vtkFileBase,onTheFlyRefinement=False, maxOrbitals=None, maxSCFIterations=None):
    
    tree.E = -1.0 # set initial energy guess


    numberOfTargets = tree.numberOfGridpoints                # set N to be the number of gridpoints.  These will be all the targets
    greenIterations_KohnSham_SCF(tree, scfTolerance, energyTolerance, numberOfTargets, gradientFree, mixingScheme, mixingParameter, subtractSingularity, 
                                smoothingN, smoothingEps,inputFile=inputFile,outputFile=outputFile, 
                                onTheFlyRefinement=onTheFlyRefinement, vtkExport=vtkExport, maxOrbitals=maxOrbitals, maxSCFIterations=maxSCFIterations)

#     greenIterations_KohnSham_SINGSUB(tree, scfTolerance, energyTolerance, numberOfTargets, subtractSingularity, 
#                                 smoothingN, smoothingEps,auxiliaryFile=auxiliaryFile, 
#                                 onTheFlyRefinement=onTheFlyRefinement, vtkExport=vtkExport)


    header = ['domainSize','minDepth','maxDepth','order','numberOfCells','numberOfPoints','gradientFree',
              'divideCriterion','divideParameter','energyTolerance',
              'GreenSingSubtracted', 'orbitalEnergies', 'BandEnergy', 'KineticEnergy',
              'ExchangeEnergy','CorrelationEnergy','ElectrostaticEnergy','TotalEnergy']
    
    myData = [domainSize,tree.minDepthAchieved,tree.maxDepthAchieved,tree.px,tree.numberOfCells,tree.numberOfGridpoints,gradientFree,
              divideCriterion,divideParameter,energyTolerance,
              subtractSingularity,
              tree.orbitalEnergies-tree.gaugeShift, tree.totalBandEnergy, tree.totalKinetic, tree.totalEx, tree.totalEc, tree.totalElectrostatic, tree.E]
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

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']

    print('='*70)
    print('='*70)
    print('='*70,'\n')
    
    """ Normal Run """
#     tree = setUpTree()
#     startTime = timer()
#     tree = setUpTree()
#     testGreenIterationsGPU(tree,vtkExport=False,onTheFlyRefinement=False, maxSCFIterations=1)
    


    """ Refinement based on deepest state """
#     tree = setUpTree(onlyFillOne=True)  
    tree = setUpTree()  
    
#     testGreenIterationsGPU(tree,vtkExport=False,onTheFlyRefinement=False, maxOrbitals=1, maxSCFIterations=1)
    testGreenIterationsGPU(tree,vtkExport=False,onTheFlyRefinement=False)
#     testGreenIterationsGPU(tree,vtkExport=False,onTheFlyRefinement=False, maxSCFIterations=1)
    
#     
#     print('\n\n\n\nNow refine based on errors in each cell, the re-initialize: ')
#     tree.compareToReferenceEnergies(refineFraction = 0.1 )
#     tree = updateTree(tree,onlyFillOne=True) 
#     testGreenIterationsGPU(tree,vtkExport=False,onTheFlyRefinement=False, maxOrbitals=1, maxSCFIterations=1)
#     
# #     print('\n\n\n\nNow refine based on errors in each cell, the re-initialize: ')
# #     tree.compareToReferenceEnergies(refineFraction = 0.05 )
# #     tree = updateTree(tree,onlyFillOne=True) 
# #     testGreenIterationsGPU(tree,vtkExport=False,onTheFlyRefinement=False, maxOrbitals=1, maxSCFIterations=1)
# #     
# #     print('\n\n\n\nNow refine based on errors in each cell, the re-initialize: ')
# #     tree.compareToReferenceEnergies(refineFraction = 0.05 )
# #     tree = updateTree(tree,onlyFillOne=True) 
# #     testGreenIterationsGPU(tree,vtkExport=False,onTheFlyRefinement=False, maxOrbitals=1, maxSCFIterations=1)
# #     
# #     print('\n\n\n\nNow refine based on errors in each cell, the re-initialize: ')
# #     tree.compareToReferenceEnergies(refineFraction = 0.05 )
# #     tree = updateTree(tree,onlyFillOne=True) 
# #     testGreenIterationsGPU(tree,vtkExport=False,onTheFlyRefinement=False, maxOrbitals=1, maxSCFIterations=1)
#     
#     print('Now do the real thing...')
# #     tree.compareToReferenceEnergies(refineFraction = 0.05 )
#     tree = updateTree(tree) 
#     testGreenIterationsGPU(tree,vtkExport=False,onTheFlyRefinement=False, maxSCFIterations=None)
#     
#     