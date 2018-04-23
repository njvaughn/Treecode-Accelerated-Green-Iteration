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
N_elements      = float(sys.argv[4])
# divideTol1      = float(sys.argv[4])
# divideTol2      = float(sys.argv[5]) ### set energyResidual argv to 6
energyResidual  = float(sys.argv[5])


def setUpTree():
    '''
    setUp() gets called before every test below.
    '''
    xmin = ymin = zmin = -domainSize
    xmax = ymax = zmax = domainSize
    tree = Tree(xmin,xmax,ymin,ymax,zmin,zmax)
#     tree.buildTree( minLevels=minDepth, maxLevels=maxDepth, divideTolerance1=divideTol1, divideTolerance2=divideTol2, printTreeProperties=True)
    tree.buildTree( minLevels=minDepth, maxLevels=maxDepth, N=N_elements, printTreeProperties=True)
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

    print('\nComputing Ground State energy...')
    tree.normalizeWavefunction()  
    tree.computeKineticOnList()
    tree.computePotentialOnList(epsilon=0.0)
    energyErrorGS_analyticPsi = -0.5-tree.totalKinetic-tree.totalPotential
    print('\nGround State Energy:            %.6g Hartree' %float((tree.totalKinetic+tree.totalPotential)))
    print(  'Ground State Error:             %.6g mHartree' %float(energyErrorGS_analyticPsi*1000.0))
 
 
    
    tree.populatePsiWithAnalytic(1)
    testPoint = tree.root.children[1,1,1].gridpoints[1,1,1]
    excitedStateMultiplicativeFactor = testPoint.psi / trueWavefunction(1, testPoint.x, testPoint.y, testPoint.z)
#         print('Excited state normalization factor ', excitedStateMultiplicativeFactor) 
    print('\nComputing Excited State energy...')
    tree.normalizeWavefunction()  
    tree.computeKineticOnList()
    tree.computePotentialOnList(epsilon=0.0)
    energyErrorFES_analyticPsi = -0.125-tree.totalKinetic-tree.totalPotential
    print('\nExcited State Energy:            %.6g Hartree' %float((tree.totalKinetic+tree.totalPotential)))
    print(  'Excited State Error:             %.6g mHartree' %float(energyErrorFES_analyticPsi*1000.0))
 
    
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
#     for element in tree.masterList:
#         for i,j,k in ThreeByThreeByThree:
#             element[1].gridpoints[i,j,k].setAnalyticPsi(0)
#             element[1].gridpoints[i,j,k].psi *= groundStateMultiplicativeFactor
#     tree.copyPsiToFinalWavefunction(0)
#      
#     print('='*70,'\n')
     
#     # reset initial guesses
#     tree.E = -0.25 # set initial energy guess
#     for element in tree.masterList:            
#         for i,j,k in ThreeByThreeByThree:
# #                 element[1].gridpoints[i,j,k].setAnalyticPsi(2)
#             element[1].gridpoints[i,j,k].setPsi(np.random.rand(1))
#              
#              
#     greenIterations(tree, 1, energyResidual, N, normalizationFactor=excitedStateMultiplicativeFactor, visualize=False)
#     tree.copyPsiToFinalWavefunction(1)
#     Etrue = -0.125
#     energyErrorFES  = tree.E-Etrue
#     psiL2ErrorFES   = tree.L2NormError
#     psiLinfErrorFES = tree.maxCellError
    
#     testFunction = 'sum(psi_i**4)'
#     testFunction = 'PsiGS*V*PsiGS'
#     testFunction2 = 'PsiGS^2*Volume'
#     testFunction = 'PsiGS'
    testFunction1 = 'LevineWilkinsOrder1'
#     testFunction1 = 'Uniform'
#     testFunction1 = 'psiVariation'
#     testFunction2 = 'max(Volume/r^3)'
#     testFunction2 = 'psi*dx'
#     testFunction = '1/r^2'
#     testFunction = 'Potential^2*Volume'

    
    myData = [domainSize,tree.minDepthAchieved,tree.maxDepthAchieved,tree.numberOfCells,tree.numberOfGridpoints,
              testFunction1,N_elements,energyResidual,
              energyErrorGS_analyticPsi,energyErrorGS,psiL2ErrorGS,psiLinfErrorGS,
              energyErrorFES_analyticPsi,energyErrorFES,psiL2ErrorFES,psiLinfErrorFES]
#     myData = [domainSize,tree.minDepthAchieved,tree.maxDepthAchieved,tree.numberOfGridpoints,testFunction1,divideTol1,testFunction2,divideTol2,energyResidual,
#               energyErrorGS,psiL2ErrorGS,psiLinfErrorGS,
#               energyErrorFES,psiL2ErrorFES,psiLinfErrorFES]
 
    myFile = open('/home/njvaughn/results/accuracyResults.csv', 'a')
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

#     print("\n\nRun Parameters: \n"
#           "Domain Size:                 %.1f \n"
#           "Divide Tolerance1:           %1.2e \n"
#           "Divide Tolerance2:           %1.2e \n"
#           "Minimum Depth                %i levels \n"
#           "Maximum Depth:               %i levels \n"
#           "Energy Residual:             %.3g seconds." 
#           %(domainSize, divideTol1, divideTol2, tree.minDepthAchieved,tree.maxDepthAchieved,energyResidual))
#     

    print("\n\nRun Parameters: \n"
          "Domain Size:                 %.1f \n"
          "Mesh Density Number:         %1.2e \n"
          "Minimum Depth                %i levels \n"
          "Maximum Depth:               %i levels \n"
          "Energy Residual:             %.3g seconds." 
          %(domainSize, N_elements, tree.minDepthAchieved,tree.maxDepthAchieved,energyResidual))
    
    print('\nTotal Time: %f seconds' %(timer()-startTime) )
    print('='*70)
    print('='*70)
    print('='*70,'\n')

        
    
    
    
    
    
