'''
Created on Mar 10, 2018

@author: nathanvaughn
'''
import unittest
from TreeStruct import Tree
from CellStruct import Cell
from GridpointStruct import GridPoint
from hydrogenPotential import trueWavefunction


class Test(unittest.TestCase):


    @classmethod
    def setUpClass(self):
        '''
        Generate a mesh on [-1,1]x[-1,1]x[-1,1] containing 8 total cells for testing.  
        setUp() gets called before every test below.
        '''
        print("Default tree used for all tests except where a new test-specific tree is built")
        self.xmin = self.ymin = self.zmin = -10
        self.xmax = self.ymax = self.zmax = 10
        self.tree = Tree(self.xmin,self.xmax,self.ymin,self.ymax,self.zmin,self.zmax)
        self.tree.buildTree( minLevels=0, maxLevels=2, divideTolerance=0.00125, printTreeProperties=True)
        self.tree.E = -0.5


    def testConvolution(self):
        self.tree.GreenFunctionConvolution(timeConvolution=True)
        print('Convolution took         %.4f seconds. ' %self.tree.ConvolutionTime)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()