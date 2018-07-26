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
from greenIterations import greenIterations_KohnSham_H2

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
energyResidual      = float(sys.argv[10])
coordinateFile      = str(sys.argv[11])
outFile             = str(sys.argv[12])
vtkFileBase         = str(sys.argv[13])


def setUpTree():
    '''
    setUp() gets called before every test below.
    '''
    xmin = ymin = zmin = -domainSize
    xmax = ymax = zmax = domainSize
    tree = Tree(xmin,xmax,order,ymin,ymax,order,zmin,zmax,order,nElectrons=2,nOrbitals=1,coordinateFile=coordinateFile)
#         def __init__(self, xmin,xmax,px,ymin,ymax,py,zmin,zmax,pz,coordinateFile,numberOfStates=1,xcFunctional="LDA_XC_LP_A",polarization="unpolarized"):

#     tree.buildTree( minLevels=minDepth, maxLevels=maxDepth, divideTolerance1=divideTol1, divideTolerance2=divideTol2, printTreeProperties=True)
    print('max depth ', maxDepth)
    tree.buildTree( minLevels=minDepth, maxLevels=maxDepth, divideCriterion=divideCriterion, divideParameter=divideParameter, printTreeProperties=True)
#     for element in tree.masterList:
#         
# #             element[1].gridpoints[1,1,1].setPsi(np.random.rand(1))
#         for i,j,k in tree.PxByPyByPz:
#             element[1].gridpoints[i,j,k].setPsi(np.random.rand(1))
            
    return tree
    
    
def testGreenIterationsGPU(tree,vtkExport=vtkFileBase,onTheFlyRefinement=False):
    
    
    # get normalization factors for finite domain analytic waves
#     tree.populatePhi()
#     tree.updateDensityAtQuadpoints()
    
#     groundStateMultiplicativeFactor = testPoint.phi / trueWavefunction(0, testPoint.x, testPoint.y, testPoint.z)  


    tree.E = -1.0 # set initial energy guess


    numberOfTargets = tree.numberOfGridpoints                # set N to be the number of gridpoints.  These will be all the targets
    greenIterations_KohnSham_H2(tree, 0, energyResidual, numberOfTargets, subtractSingularity, 
                                smoothingN, smoothingEps, 
                                normalizationFactor=1.0,
                                onTheFlyRefinement=onTheFlyRefinement, vtkExport=vtkExport)


    header = ['domainSize','minDepth','maxDepth','order','numberOfCells','numberOfPoints',
              'divideCriterion','divideParameter','energyResidual',
              'GreenSingSubtracted', 
              'computedE', 'computedHOMO', 'errorE','errorHOMO']
    
    myData = [domainSize,tree.minDepthAchieved,tree.maxDepthAchieved,tree.px,tree.numberOfCells,tree.numberOfGridpoints,
              divideCriterion,divideParameter,energyResidual,
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
    
    