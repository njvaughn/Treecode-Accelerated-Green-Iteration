<<<<<<< HEAD
'''
Created on Apr 24, 2018

@author: nathanvaughn
'''
import sys
sys.path.append('../src/dataStructures')
sys.path.append('../src/utilities')

import unittest
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
        self.tree.buildTree( minLevels=3, maxLevels=20, divideCriterion='LW1', divideParameter=100, printTreeProperties=True)
        self.tree.populatePsiWithAnalytic(0)
        
        
    def testMidpointBeforeAndAfter(self):
        before = self.tree.root.children[1,1,1].children[0,1,0].gridpoints[1,1,1]
        beforePsi = before.psi
        
#         sources = tree.extractLeavesMidpointsOnly()  # extract the source point locations.  Currently, these are just all the leaf midpoints
        targets = self.tree.extractLeavesAllGridpoints()  # extract the target point locations.  Currently, these are all 27 gridpoints per cell (no redundancy)
        psiNew = targets[:,3]
#         print(targets.shape)
#         print(psiNew.shape)
        
        
        self.tree.importPsiOnLeaves(psiNew)
        after = self.tree.root.children[1,1,1].children[0,1,0].gridpoints[1,1,1]
        afterPsi = after.psi
        self.assertEqual(before, after , "Error when exporting then importing same array. ")
        self.assertEqual(beforePsi, afterPsi , "Error when exporting then importing same array. ")



    def testBoundaryPointBeforeAndAfter(self):
        before = self.tree.root.children[1,1,1].children[0,1,0].gridpoints[0,2,0]
        beforePsi = before.psi
        
        targets = self.tree.extractLeavesAllGridpoints()  # extract the target point locations.  Currently, these are all 27 gridpoints per cell (no redundancy)
        psiNew = targets[:,3]
        
        self.tree.importPsiOnLeaves(psiNew)
        after = self.tree.root.children[1,1,1].children[0,1,0].gridpoints[0,2,0]
        afterPsi = after.psi
        self.assertEqual(before, after , "Error when exporting then importing same array. ")
        self.assertEqual(beforePsi, afterPsi , "Error when exporting then importing same array. ")



if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
=======
'''
Created on Apr 24, 2018

@author: nathanvaughn
'''
import sys
sys.path.append('../src/dataStructures')
sys.path.append('../src/utilities')

import unittest
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
        self.tree.buildTree( minLevels=3, maxLevels=20, divideCriterion='LW1', divideParameter=100, printTreeProperties=True)
        self.tree.populatePsiWithAnalytic(0)
        
        
    def testMidpointBeforeAndAfter(self):
        before = self.tree.root.children[1,1,1].children[0,1,0].gridpoints[1,1,1]
        beforePsi = before.psi
        
#         sources = tree.extractLeavesMidpointsOnly()  # extract the source point locations.  Currently, these are just all the leaf midpoints
        targets = self.tree.extractLeavesAllGridpoints()  # extract the target point locations.  Currently, these are all 27 gridpoints per cell (no redundancy)
        psiNew = targets[:,3]
#         print(targets.shape)
#         print(psiNew.shape)
        
        
        self.tree.importPsiOnLeaves(psiNew)
        after = self.tree.root.children[1,1,1].children[0,1,0].gridpoints[1,1,1]
        afterPsi = after.psi
        self.assertEqual(before, after , "Error when exporting then importing same array. ")
        self.assertEqual(beforePsi, afterPsi , "Error when exporting then importing same array. ")



    def testBoundaryPointBeforeAndAfter(self):
        before = self.tree.root.children[1,1,1].children[0,1,0].gridpoints[0,2,0]
        beforePsi = before.psi
        
        targets = self.tree.extractLeavesAllGridpoints()  # extract the target point locations.  Currently, these are all 27 gridpoints per cell (no redundancy)
        psiNew = targets[:,3]
        
        self.tree.importPsiOnLeaves(psiNew)
        after = self.tree.root.children[1,1,1].children[0,1,0].gridpoints[0,2,0]
        afterPsi = after.psi
        self.assertEqual(before, after , "Error when exporting then importing same array. ")
        self.assertEqual(beforePsi, afterPsi , "Error when exporting then importing same array. ")



if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
>>>>>>> refs/remotes/eclipse_auto/master
    unittest.main()