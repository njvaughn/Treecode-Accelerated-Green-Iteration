'''
testUniformRefinement.py
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
outputFileBase          = str(sys.argv[12])
inputFile           = str(sys.argv[13])

# coordinateFile      = str(sys.argv[12])
# auxiliaryFile      = str(sys.argv[13])
# nElectrons          = int(sys.argv[14])
# nOrbitals          = int(sys.argv[15])
# outFile             = str(sys.argv[16])
# vtkFileBase         = str(sys.argv[17])
vtkFileBase='/home/njvaughn/results_CO/orbitals'

def setUpTree():
    '''
    setUp() gets called before every test below.
    '''
    xmin = ymin = zmin = -domainSize
    xmax = ymax = zmax = domainSize

    [coordinateFile, DummyOutputFile] = np.genfromtxt(inputFile,dtype="|U100")[:2]
    [nElectrons, nOrbitals, Eband, Ekinetic, Eexchange, Ecorrelation, Eelectrostatic, Etotal, gaugeShift] = np.genfromtxt(inputFile)[2:]
    nElectrons = int(nElectrons)
    nOrbitals = int(nOrbitals)
    

    print([coordinateFile, outputFileBase, nElectrons, nOrbitals, 
     Etotal, Eexchange, Ecorrelation, Eband, gaugeShift])
    tree = Tree(xmin,xmax,order,ymin,ymax,order,zmin,zmax,order,nElectrons,nOrbitals,gaugeShift=gaugeShift,
                coordinateFile=coordinateFile,inputFile=inputFile)#, iterationOutFile=outputFile)

    
    
    print('max depth ', maxDepth)
    tree.buildTree( minLevels=minDepth, maxLevels=maxDepth, initializationType='atomic',divideCriterion=divideCriterion, divideParameter=divideParameter, printTreeProperties=True)


    tree.normalizeDensity()
    return tree
    
    
def testUniformRefinement(tree,refinementLevels=3,R=None,vtkExport=vtkFileBase,onTheFlyRefinement=False):
    
    
#     for i in range(refinementLevels):
    refinementCounter = 0
    while refinementCounter < refinementLevels:
        
        
        tree.E = -1.0 # set initial energy guess
        outputFile =  outputFileBase + str(tree.numberOfCells) + '.csv'
    
        numberOfTargets = tree.numberOfGridpoints                # set N to be the number of gridpoints.  These will be all the targets
        greenIterations_KohnSham_SCF(tree, scfTolerance, energyTolerance, numberOfTargets, subtractSingularity, 
                                    smoothingN, smoothingEps,inputFile=inputFile,outputFile=outputFile, 
                                    onTheFlyRefinement=onTheFlyRefinement, vtkExport=vtkExport)
    
    #     greenIterations_KohnSham_SINGSUB(tree, scfTolerance, energyTolerance, numberOfTargets, subtractSingularity, 
    #                                 smoothingN, smoothingEps,auxiliaryFile=auxiliaryFile, 
    #                                 onTheFlyRefinement=onTheFlyRefinement, vtkExport=vtkExport)
    
    
        header = ['domainSize','minDepth','maxDepth','order','numberOfCells','numberOfPoints',
                  'divideCriterion','divideParameter','energyTolerance',
                  'GreenSingSubtracted', 
                  'orbitalEnergies', 'ExchangePotential', 'CorrelationPotential','BandEnergy','ExchangeEnergy','CorrelationEnergy','TotalEnergy']
        
        myData = [domainSize,tree.minDepthAchieved,tree.maxDepthAchieved,tree.px,tree.numberOfCells,tree.numberOfGridpoints,
                  divideCriterion,divideParameter,energyTolerance,
                  subtractSingularity,
                  tree.orbitalEnergies, tree.totalVx, tree.totalVc, 
                          tree.totalBandEnergy, tree.totalEx, tree.totalEc, tree.E]
    #               tree.E, tree.
    #               tree.E, tree.orbitalEnergies[0], abs(tree.E+1.1373748), abs(tree.orbitalEnergies[0]+0.378665)]
        
    
        runComparisonFile = '/home/njvaughn/OxygenResults/runComparison.csv'
        if not os.path.isfile(runComparisonFile):
            myFile = open(runComparisonFile, 'a')
            with myFile:
                writer = csv.writer(myFile)
                writer.writerow(header) 
            
        
        myFile = open(runComparisonFile, 'a')
        with myFile:
            writer = csv.writer(myFile)
            writer.writerow(myData)
            
            
        refinementCounter+=1
        if refinementCounter < refinementLevels:
            # not done yet, need to refine again
            if not R:
                print('Uniformly refining whole domain')
                tree.uniformlyRefine()
            else:
                print('Uniformly refining inside radius %1.2f' %R)
                tree.uniformlyRefineWithinRadius(R)
                
                
def testManualRefinement(tree,refinementLevels,R,vtkExport=vtkFileBase,onTheFlyRefinement=False):


#     for i in range(refinementLevels):
    tree.uniformlyRefineWithinRadius(R)
#     tree.uniformlyRefineWithinRadius(R/2)
#     tree.uniformlyRefineWithinRadius(R/4)
    refinementCounter = 0
    while refinementCounter < refinementLevels:
        
        
        tree.E = -1.0 # set initial energy guess
        outputFile =  outputFileBase + str(tree.numberOfCells) + '.csv'
    
        numberOfTargets = tree.numberOfGridpoints                # set N to be the number of gridpoints.  These will be all the targets
        greenIterations_KohnSham_SCF(tree, scfTolerance, energyTolerance, numberOfTargets, subtractSingularity, 
                                    smoothingN, smoothingEps,inputFile=inputFile,outputFile=outputFile, 
                                    onTheFlyRefinement=onTheFlyRefinement, vtkExport=vtkExport)
    
    #     greenIterations_KohnSham_SINGSUB(tree, scfTolerance, energyTolerance, numberOfTargets, subtractSingularity, 
    #                                 smoothingN, smoothingEps,auxiliaryFile=auxiliaryFile, 
    #                                 onTheFlyRefinement=onTheFlyRefinement, vtkExport=vtkExport)
    
    
        header = ['domainSize','minDepth','maxDepth','order','numberOfCells','numberOfPoints',
              'divideCriterion','divideParameter','energyTolerance',
              'GreenSingSubtracted', 'orbitalEnergies', 'BandEnergy', 'KineticEnergy',
              'ExchangeEnergy','CorrelationEnergy','ElectrostaticEnergy','TotalEnergy']
    
        myData = [domainSize,tree.minDepthAchieved,tree.maxDepthAchieved,tree.px,tree.numberOfCells,tree.numberOfGridpoints,
              divideCriterion,divideParameter,energyTolerance,
              subtractSingularity,
              tree.orbitalEnergies-tree.gaugeShift, tree.totalBandEnergy, tree.totalKinetic, tree.totalEx, tree.totalEc, tree.totalElectrostatic, tree.E]
    #               tree.E, tree.
    #               tree.E, tree.orbitalEnergies[0], abs(tree.E+1.1373748), abs(tree.orbitalEnergies[0]+0.378665)]
        
        runComparisonFile = os.path.split(outputFile)[0] + '/runComparison.csv'
    
#         runComparisonFile = '/home/njvaughn/OxygenFirstSCF/runComparison_manualRefine.csv'
        if not os.path.isfile(runComparisonFile):
            myFile = open(runComparisonFile, 'a')
            with myFile:
                writer = csv.writer(myFile)
                writer.writerow(header) 
            
        
        myFile = open(runComparisonFile, 'a')
        with myFile:
            writer = csv.writer(myFile)
            writer.writerow(myData)
            
            
        refinementCounter+=1
        if refinementCounter < refinementLevels:
            # not done yet, need to refine again
            print('Uniformly refining inside radius %1.2f' %R)
            tree.uniformlyRefineWithinRadius(R)
                      
              
    
    


    


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']

    print('='*70)
    print('='*70)
    print('='*70,'\n')
    startTime = timer()
    tree = setUpTree()
    
#     testUniformRefinement(tree,refinementLevels=2,R=None,vtkExport=False,onTheFlyRefinement=False)
    testManualRefinement(tree,refinementLevels=1,R=0.05,vtkExport=False,onTheFlyRefinement=False)
    
    