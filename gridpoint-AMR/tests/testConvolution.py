'''
Created on Mar 10, 2018

@author: nathanvaughn
'''
import unittest
import numpy as np
from TreeStruct import Tree
from CellStruct import Cell
from GridpointStruct import GridPoint
from hydrogenPotential import trueWavefunction


class Test(unittest.TestCase):


    @classmethod
    def setUpClass(self):
        '''
        setUp() gets called before every test below.
        '''
        self.xmin = self.ymin = self.zmin = -10
        self.xmax = self.ymax = self.zmax = 10
        self.tree = Tree(self.xmin,self.xmax,self.ymin,self.ymax,self.zmin,self.zmax)
        self.tree.buildTree( minLevels=3, maxLevels=7, divideTolerance=0.2, printTreeProperties=True)
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
        
        for i in range(15):
#             print('Performing Convolution using masterList')
            self.tree.GreenFunctionConvolutionList(timeConvolution=True)
#             print('Convolution took         %.4f seconds. ' %self.tree.ConvolutionTime)
            self.tree.computeWaveErrors()
            print('Convolution wavefunction errors: %.3e L2,  %.3e max' %(self.tree.L2NormError, self.tree.maxCellError))
        self.tree.visualizeMesh('psi')
       
#         print('Performing Convolution using masterList')
#         self.tree.GreenFunctionConvolutionList(timeConvolution=True)
#         print('Convolution took         %.4f seconds. ' %self.tree.ConvolutionTime)
#         self.tree.computeWaveErrors()
#         print('Second Convolution wavefunction errors: %.3e L2,  %.3e max' %(self.tree.L2NormError, self.tree.maxCellError))
#         self.tree.visualizeMesh('psi')
#         
#         print('Performing Convolution using masterList')
#         self.tree.GreenFunctionConvolutionList(timeConvolution=True)
#         print('Convolution took         %.4f seconds. ' %self.tree.ConvolutionTime)
#         self.tree.computeWaveErrors()
#         print('Third Convolution wavefunction errors: %.3e L2,  %.3e max' %(self.tree.L2NormError, self.tree.maxCellError))
#         self.tree.visualizeMesh('psi')
#         
#         print('Performing Convolution using masterList')
#         self.tree.GreenFunctionConvolutionList(timeConvolution=True)
#         print('Convolution took         %.4f seconds. ' %self.tree.ConvolutionTime)
#         self.tree.computeWaveErrors()
#         print('Fourth Convolution wavefunction errors: %.3e L2,  %.3e max' %(self.tree.L2NormError, self.tree.maxCellError))
#         self.tree.visualizeMesh('psi')
#         
#         print('Performing Convolution using masterList')
#         self.tree.GreenFunctionConvolutionList(timeConvolution=True)
#         print('Convolution took         %.4f seconds. ' %self.tree.ConvolutionTime)
#         self.tree.computeWaveErrors()
#         print('Fifth Convolution wavefunction errors: %.3e L2,  %.3e max' %(self.tree.L2NormError, self.tree.maxCellError))
#         self.tree.visualizeMesh('psi')


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()