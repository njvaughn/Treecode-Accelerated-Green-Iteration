'''
Created on Feb 23, 2018

@author: nathanvaughn
'''
import unittest
from dataStructs import Tree
from TwoDNeighborCheck import *

class Test(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.xmin = self.ymin = self.zmin = -10
        self.xmax = self.ymax = self.zmax = 10
        self.tree = Tree(self.xmin,self.xmax,self.ymin,self.ymax,self.zmin,self.zmax)
        self.tree.buildTree( minLevels=0, maxLevels=4, divideTolerance=0.025)
#         self.tree.walkTree()
        self.tree.visualizeMesh(attributeForColoring='volume')



    def test(self):
        pass


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()