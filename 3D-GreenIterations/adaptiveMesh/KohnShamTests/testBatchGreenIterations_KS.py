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
energyTolerance      = float(sys.argv[10])
scfTolerance        = float(sys.argv[11])
coordinateFile      = str(sys.argv[12])
auxiliaryFile      = str(sys.argv[13])
nElectrons          = int(sys.argv[14])
nOrbitals          = int(sys.argv[15])
outFile             = str(sys.argv[16])
vtkFileBase         = str(sys.argv[17])


def setUpTree():
    '''
    setUp() gets called before every test below.
    '''
    xmin = ymin = zmin = -domainSize
    xmax = ymax = zmax = domainSize
    tree = Tree(xmin,xmax,order,ymin,ymax,order,zmin,zmax,order,nElectrons,nOrbitals,
                coordinateFile=coordinateFile,auxiliaryFile=auxiliaryFile)

    print('max depth ', maxDepth)
    tree.buildTree( minLevels=minDepth, maxLevels=maxDepth, initializationType='atomic',divideCriterion=divideCriterion, divideParameter=divideParameter, printTreeProperties=True)
#     for element in tree.masterList:
#         
# #             element[1].gridpoints[1,1,1].setPsi(np.random.rand(1))
#         for i,j,k in tree.PxByPyByPz:
#             element[1].gridpoints[i,j,k].setPsi(np.random.rand(1))
            
    return tree
    
    
def testGreenIterationsGPU(tree,vtkExport=vtkFileBase,onTheFlyRefinement=False):
    
    tree.E = -1.0 # set initial energy guess


    numberOfTargets = tree.numberOfGridpoints                # set N to be the number of gridpoints.  These will be all the targets
    greenIterations_KohnSham_SCF(tree, scfTolerance, energyTolerance, numberOfTargets, subtractSingularity, 
                                smoothingN, smoothingEps,auxiliaryFile=auxiliaryFile, 
                                onTheFlyRefinement=onTheFlyRefinement, vtkExport=vtkExport)

#     greenIterations_KohnSham_SINGSUB(tree, scfTolerance, energyTolerance, numberOfTargets, subtractSingularity, 
#                                 smoothingN, smoothingEps,auxiliaryFile=auxiliaryFile, 
#                                 onTheFlyRefinement=onTheFlyRefinement, vtkExport=vtkExport)


    header = ['domainSize','minDepth','maxDepth','order','numberOfCells','numberOfPoints',
              'divideCriterion','divideParameter','energyTolerance',
              'GreenSingSubtracted', 
              'computedE', 'computedHOMO', 'errorE','errorHOMO']
    
    myData = [domainSize,tree.minDepthAchieved,tree.maxDepthAchieved,tree.px,tree.numberOfCells,tree.numberOfGridpoints,
              divideCriterion,divideParameter,energyTolerance,
              subtractSingularity,
              tree.E, tree.orbitalEnergies[0], abs(tree.E+1.1373748), abs(tree.orbitalEnergies[0]+0.378665)]
    

    if not os.path.isfile(outFile):
        myFile = open(outFile, 'a')
        with myFile:
            writer = csv.writer(myFile)
            writer.writerow(header) 
        
    
    myFile = open(outFile, 'a')
    with myFile:
        writer = csv.writer(myFile)
        writer.writerow(myData)    
    


    


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']

    print('='*70)
    print('='*70)
    print('='*70,'\n')
    startTime = timer()
    tree = setUpTree()
#     testGreenIterationsGPU(tree,vtkExport=vtkFile)
#     testGreenIterationsGPU(tree,vtkExport=vtkFileBase,onTheFlyRefinement=False)
    testGreenIterationsGPU(tree,vtkExport=False,onTheFlyRefinement=False)
    
    