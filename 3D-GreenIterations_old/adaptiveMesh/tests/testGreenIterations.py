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

from TreeStruct import Tree
from convolution import gpuConvolution, greenIterations
from hydrogenPotential import trueWavefunction

ThreeByThreeByThree = [element for element in itertools.product(range(3),range(3),range(3))]

class TestGreenIterations(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        '''
        setUp() gets called before every test below.
        '''
        self.xmin = self.ymin = self.zmin = -10
        self.xmax = self.ymax = self.zmax = 10
        self.tree = Tree(self.xmin,self.xmax,self.ymin,self.ymax,self.zmin,self.zmax)
        self.tree.buildTree( minLevels=4, maxLevels=8, divideTolerance=0.04, printTreeProperties=True)
        for element in self.tree.masterList:
            
#             element[1].gridpoints[1,1,1].setPsi(np.random.rand(1))
            for i,j,k in ThreeByThreeByThree:
                element[1].gridpoints[i,j,k].setPsi(np.random.rand(1))
        


    
    @unittest.skip('Skipped CPU convolution')    
    def testGreenIterations(self):
        # get normalization factors for finite domain analytic waves
        self.tree.populatePsiWithAnalytic(0)
        testPoint = self.tree.root.children[1,1,1].gridpoints[1,1,1]
        groundStateMultiplicativeFactor = testPoint.psi / trueWavefunction(0, testPoint.x, testPoint.y, testPoint.z)   
        
        self.tree.populatePsiWithAnalytic(1)
        testPoint = self.tree.root.children[1,1,1].gridpoints[1,1,1]
        excitedStateMultiplicativeFactor = testPoint.psi / trueWavefunction(1, testPoint.x, testPoint.y, testPoint.z)
        
        
        self.tree.E = -1.0 # initial guess
          
        for i in range(1):
            print()
            self.tree.GreenFunctionConvolutionList(timeConvolution=True)
            print('Convolution took:                %.4f seconds. ' %self.tree.ConvolutionTime)
            self.tree.computeWaveErrors()
            print('Convolution wavefunction errors: %.3e L2,  %.3e max' %(self.tree.L2NormError, self.tree.maxCellError))
            self.tree.updateEnergy()
            print('Kinetic Energy:                  %.3f ' %self.tree.totalKinetic)
            print('Potential Energy:               %.3f ' %self.tree.totalPotential)
            print('Updated Energy Value:            %.3f Hartree, %.3e error' %(self.tree.E, self.tree.E+0.5))
    
    def testGreenIterationsGPU(self):
        
        
        # get normalization factors for finite domain analytic waves
        self.tree.populatePsiWithAnalytic(0)
        testPoint = self.tree.root.children[1,1,1].gridpoints[1,1,1]
        groundStateMultiplicativeFactor = testPoint.psi / trueWavefunction(0, testPoint.x, testPoint.y, testPoint.z)  
        print('Ground state normalization factor ', groundStateMultiplicativeFactor) 
        
        self.tree.populatePsiWithAnalytic(1)
        testPoint = self.tree.root.children[1,1,1].gridpoints[1,1,1]
        excitedStateMultiplicativeFactor = testPoint.psi / trueWavefunction(1, testPoint.x, testPoint.y, testPoint.z)
        print('Excited state normalization factor ', excitedStateMultiplicativeFactor) 
        
        self.tree.E = -1.0 # set initial energy guess
        
        # reset initial guesses
        for element in self.tree.masterList:            
            for i,j,k in ThreeByThreeByThree:
                element[1].gridpoints[i,j,k].setPsi(np.random.rand(1))

        N = self.tree.numberOfGridpoints                # set N to be the number of gridpoints.  These will be all the targets
        residualTolerance = 0.000005
        greenIterations(self.tree, 0, residualTolerance, N,normalizationFactor=groundStateMultiplicativeFactor,visualize=False)
        
        # set grpund state to analytic wavefunction to isolate errors in excited state
        for element in self.tree.masterList:
            for i,j,k in ThreeByThreeByThree:
                element[1].gridpoints[i,j,k].setAnalyticPsi(0)
                element[1].gridpoints[i,j,k].psi *= groundStateMultiplicativeFactor
        self.tree.copyPsiToFinalWavefunction(0)
         
        testGridPoint = self.tree.masterList[356][1].gridpoints[0,1,2]
#         print(testGridPoint.finalWavefunction)
        self.assertEqual(testGridPoint.psi, testGridPoint.finalWavefunction[0], "Psi did not get copied to first element of finalWavefunction")
        print('='*70,'\n')
         
        # reset initial guesses
        self.tree.E = -0.25 # set initial energy guess
        for element in self.tree.masterList:            
            for i,j,k in ThreeByThreeByThree:
#                 element[1].gridpoints[i,j,k].setAnalyticPsi(2)
                element[1].gridpoints[i,j,k].setPsi(np.random.rand(1))
                 
                 
        greenIterations(self.tree, 1, residualTolerance, N, normalizationFactor=excitedStateMultiplicativeFactor, visualize=True)
        self.tree.copyPsiToFinalWavefunction(1)
#         



if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
    
    
    
    
    