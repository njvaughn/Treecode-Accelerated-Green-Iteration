'''
Created on Mar 5, 2018

@author: nathanvaughn
'''
import unittest


from Tree import Tree

class TestEnergyComputation(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.xmin = self.ymin = self.zmin = -10
        self.xmax = self.ymax = self.zmax = 10
        self.tree = Tree(self.xmin,self.xmax,self.ymin,self.ymax,self.zmin,self.zmax)
        self.tree.buildTree( minLevels=0, maxLevels=10, divideTolerance=0.005, printTreeProperties=True)



    def testPotentialComputation(self):
        self.tree.computePotentialOnTree(epsilon=0)
        print('\nPotential Error:        %.3g mHartree' %float((-1.0-self.tree.totalPotential)*1000.0))
#     @unittest.skip("Skip energy computations.")    
    def testKineticComputation(self):
        self.tree.computeKineticOnTree()
        print('\nKinetic Error:           %.3g mHartree' %float((0.5-self.tree.totalKinetic)*1000.0))
         


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()