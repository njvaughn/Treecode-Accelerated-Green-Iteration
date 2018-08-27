'''
Created on Mar 10, 2018

@author: nathanvaughn
'''
import os
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
print(ROOT_DIR)
import unittest
import numpy as np
import sys
sys.path.append('../src/dataStructures')
sys.path.append('../src/utilities')
from TreeStruct import Tree

class Test(unittest.TestCase):


    @classmethod
    def setUpClass(self):
        '''
        setUp() gets called before every test below.
        '''
        self.xmin = self.ymin = self.zmin = -10
        self.xmax = self.ymax = self.zmax = 10
        self.tree = Tree(self.xmin,self.xmax,self.ymin,self.ymax,self.zmin,self.zmax)
        self.tree.buildTree( minLevels=3, maxLevels=7, divideTolerance=0.04, printTreeProperties=True)
        for element in self.tree.masterList:
            element[1].gridpoints[1,1,1].setPsi(np.random.rand(1))
        self.tree.E = -0.5

    @unittest.skip("Skip convolution timing test for a moment")
    def testConvolution(self):
        print('\nUsing Tree')
        self.tree.GreenFunctionConvolutionRecursive(timeConvolution=True)
        print('Convolution took         %.4f seconds. ' %self.tree.ConvolutionTime)
        
        self.tree.GreenFunctionConvolutionList(timeConvolution=True)
        print('\nUsing list: ')
        print('Convolution took         %.4f seconds. ' %self.tree.ConvolutionTime)
        
    def testComputedWavefunction(self):
        self.tree.computeWaveErrors()
        print('\nInitial wavefunction errors:     %.3e L2,  %.3e max' %(self.tree.L2NormError, self.tree.maxCellError))
#         self.tree.visualizeMesh('psi')
        
        for i in range(10):
            self.tree.GreenFunctionConvolutionList(timeConvolution=True)
            print('Convolution took         %.4f seconds. ' %self.tree.ConvolutionTime)
            self.tree.computeWaveErrors()
            print('Convolution wavefunction errors: %.3e L2,  %.3e max' %(self.tree.L2NormError, self.tree.maxCellError))
        self.tree.visualizeMesh('psi')
       



if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
