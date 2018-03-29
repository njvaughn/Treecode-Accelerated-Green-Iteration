'''
testGreenIterations.py
This is a unitTest module for testing Green iterations.  It begins by building the tree-based
adaotively refined mesh, then performs Green iterations to obtain the ground state energy
and wavefunction for the single electron hydrogen atom.  -- 03/20/2018 NV

Created on Mar 13, 2018
@author: nathanvaughn
'''

import sys
sys.path.append('../src/dataStructures')
sys.path.append('../src/utilities')

import unittest
import numpy as np
from timeit import default_timer as timer
import itertools
import csv

from TreeStruct import Tree
from convolution import greenIterations
from hydrogenPotential import trueWavefunction

ThreeByThreeByThree = [element for element in itertools.product(range(3),range(3),range(3))]

domainSize      = int(sys.argv[1])
minDepth        = int(sys.argv[2])
maxDepth        = int(sys.argv[3])
divideTol       = float(sys.argv[4])
energyResidual  = float(sys.argv[5])



def setUpTree():
    '''
    setUp() gets called before every test below.
    '''
    xmin = ymin = zmin = -domainSize
    xmax = ymax = zmax = domainSize
    tree = Tree(xmin,xmax,ymin,ymax,zmin,zmax)
    tree.buildTree( minLevels=minDepth, maxLevels=maxDepth, divideTolerance=divideTol, printTreeProperties=True)
    for element in tree.masterList:
        
#             element[1].gridpoints[1,1,1].setPsi(np.random.rand(1))
        for i,j,k in ThreeByThreeByThree:
            element[1].gridpoints[i,j,k].setPsi(np.random.rand(1))
            
    return tree
    
    
def testGreenIterationsGPU(tree):
    
    
    # get normalization factors for finite domain analytic waves
    tree.populatePsiWithAnalytic(0)
    testPoint = tree.root.children[1,1,1].gridpoints[1,1,1]
    groundStateMultiplicativeFactor = testPoint.psi / trueWavefunction(0, testPoint.x, testPoint.y, testPoint.z)  
#         print('Ground state normalization factor ', groundStateMultiplicativeFactor) 
    
    tree.populatePsiWithAnalytic(1)
    testPoint = tree.root.children[1,1,1].gridpoints[1,1,1]
    excitedStateMultiplicativeFactor = testPoint.psi / trueWavefunction(1, testPoint.x, testPoint.y, testPoint.z)
#         print('Excited state normalization factor ', excitedStateMultiplicativeFactor) 
    
    tree.E = -1.0 # set initial energy guess
    
    # reset initial guesses
    for element in tree.masterList:            
        for i,j,k in ThreeByThreeByThree:
            element[1].gridpoints[i,j,k].setPsi(np.random.rand(1))

    N = tree.numberOfGridpoints                # set N to be the number of gridpoints.  These will be all the targets
    greenIterations(tree, 0, energyResidual, N,normalizationFactor=groundStateMultiplicativeFactor,visualize=False)
    Etrue = -0.5
    energyErrorGS  = tree.E-Etrue
    psiL2ErrorGS   = tree.L2NormError
    psiLinfErrorGS = tree.maxCellError
    
    # set the ground state equal to the analytic ground state so that the ortogonalization is reliable, even if the ground state was inaccurate.  
    # Once testing is done, need to check behavior when the computed ground state is used for orthogonalization.
    for element in tree.masterList:
        for i,j,k in ThreeByThreeByThree:
            element[1].gridpoints[i,j,k].setAnalyticPsi(0)
            element[1].gridpoints[i,j,k].psi *= groundStateMultiplicativeFactor
    tree.copyPsiToFinalWavefunction(0)
     
    print('='*70,'\n')
     
    # reset initial guesses
    tree.E = -0.25 # set initial energy guess
    for element in tree.masterList:            
        for i,j,k in ThreeByThreeByThree:
#                 element[1].gridpoints[i,j,k].setAnalyticPsi(2)
            element[1].gridpoints[i,j,k].setPsi(np.random.rand(1))
             
             
    greenIterations(tree, 1, energyResidual, N, normalizationFactor=excitedStateMultiplicativeFactor, visualize=False)
    tree.copyPsiToFinalWavefunction(1)
    Etrue = -0.125
    energyErrorFES  = tree.E-Etrue
    psiL2ErrorFES   = tree.L2NormError
    psiLinfErrorFES = tree.maxCellError
    
    
    myData = [domainSize,minDepth,maxDepth,tree.numberOfGridpoints,divideTol,energyResidual,
              energyErrorGS,psiL2ErrorGS,psiLinfErrorGS,
              energyErrorFES,psiL2ErrorFES,psiLinfErrorFES]
 
    myFile = open('/home/njvaughn/Documents/GitHub/Greens-Functions-Iterative-Methods/3D-GreenIterations/adaptiveMesh/results/accuracyResults.csv', 'a')
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
    testGreenIterationsGPU(tree)
    
    
    
#     del sys.argv[1:]
#     unittest.main()
    
#     domainSize      = sys.argv[0]
# minDepth        = sys.argv[1]
# maxDepth        = sys.argv[2]
# divideTolerance = sys.argv[3]
# energyResidual  = sys.argv[4]

    print("\n\nRun Parameters: \n"
          "Domain Size:                 %.1f \n"
          "Divide Tolerance:            %1.2e \n"
          "Minimum Depth                %i levels \n"
          "Maximum Depth:               %i levels \n"
          "Energy Residual:             %.3g seconds." 
          %(domainSize, divideTol, minDepth,maxDepth,energyResidual))
    
    print('\nTotal Time: %f seconds' %(timer()-startTime) )
    print('='*70)
    print('='*70)
    print('='*70,'\n')

        
    
    
    
    
    