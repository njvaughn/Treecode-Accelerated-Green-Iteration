import unittest
import numpy as np


from dataStructs import Tree, Cell, GridPoint
from hydrogenPotential import trueWavefunction
from timer import Timer

class TestTreeStructure(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        '''
        Generate a mesh on [-1,1]x[-1,1]x[-1,1] containing 8 total cells for testing.  
        setUp() gets called before every test below.
        '''
        self.xmin = self.ymin = self.zmin = -8
        self.xmax = self.ymax = self.zmax = 8
        self.tree = Tree(self.xmin,self.xmax,self.ymin,self.ymax,self.zmin,self.zmax)
        self.tree.buildTree( minLevels=0, maxLevels=10, divideTolerance=0.000125)

    def testNeighborsPointToSameObject(self):
        '''
        Test 1:  Assert that different cells point to the same grid point object in memory.
        '''
        # check that the bottom left cell and top left cell both point to same gridpoint at their boundary
        self.assertEqual(self.tree.root.children[0,0,1].gridpoints[0,2,1], self.tree.root.children[0,1,1].gridpoints[0,0,1], 
                         "Failed Test: 00 Cell's top left gridpoint object should be the top left cell's bottom left gridpoint object")
        
    def testNeighborsFeelModifications(self):
        '''
        Test 2:  Assert that modifying a certain gridpoint in one cell is felt by all cells that are composed of that gridpoint
        '''        
        # modify a boundary gridpoint's data in bottom left cell to x=-52
        self.tree.root.children[0,0,0].gridpoints[0,2,0].x = -52
        # check that the top left cell's gridpoint data now has the updated value.
        self.assertEqual(self.tree.root.children[0,1,0].gridpoints[0,0,0].x, -52, 
                         "Failed Test:  Modified x coord of this gridpoint wasn't -52 as expected")
        
        self.tree.root.children[0,0,0].gridpoints[0,2,0].setPsi(0.55)
        self.assertEqual(self.tree.root.children[0,0,0].gridpoints[0,2,0].psi, 0.55, 
                         "Failed Test:  Psi not set properly")
        self.assertEqual(self.tree.root.children[0,1,0].gridpoints[0,0,0].psi, 0.55, 
                         "Failed Test:  Psi set from neighboring cell not noticed")
              
    def testObjectTypesAfterTreeConstruction(self):
        self.assertIsInstance(self.tree, Tree, "self is not a Tree object.")
        self.assertIsInstance(self.tree.root, Cell, "Root of tree is not a cell object.")
        self.assertIsInstance(self.tree.root.gridpoints[0,0,2], GridPoint, "Gridpoints of root are not GridPoint objects.")
        self.assertIsInstance(self.tree.root.children[0,1,0], Cell, "Children of root are not cell objects.")
        self.assertIsInstance(self.tree.root.children[0,1,0].gridpoints[2,1,2], GridPoint, "Children's gridpoints are not GridPoint objects.")
        self.assertIsInstance(self.tree.root.children[0,1,0].gridpoints[2,1,2].x, np.float, "Gridpoint data not numpy floats.")
        
    def testTreeBuildLevels(self):
        root = self.tree.root
        child = root.children[1,0,0]
        grandchild = child.children[0,1,1]
        self.assertEqual(root.level, 0, "Root level wasn't equal to 0.")
        self.assertEqual(child.level, 1, "Root's child level wasn't equal to 1.")
        self.assertEqual(grandchild.level, 2, "Root's grandchild level wasn't equal to 2.")
        self.assertEqual(root.gridpoints[0,0,0].x, self.xmin, "Root's 000 corner point doesn't have correct x value")
       
    def testTreePointersBetweenParentsAndChildren(self):
        # identify 3 generations, doesn't matter which child in each generation
        grandparent = self.tree.root
        parent = grandparent.children[0,1,1]
        child = parent.children[1,0,1]
        
        self.assertEqual(np.shape(parent.children), (2,2,2), "Shape of array of children pointers not (2,2,2)")
        self.assertEqual(parent.parent, grandparent, "Parent's parent isn't the grandparent")
        self.assertEqual(child.parent, parent, "Child's parent isn't the parent")
    
    def testAnalyticPsiSetting(self):
        self.assertEqual(self.tree.root.gridpoints[1,1,1].psi, trueWavefunction(1, 0, 0, 0), "root midpoint (the origin) doesn't have correct wavefunction.")

    def testTreeWalk(self):
#         self.tree.walkTree(attribute='divideFlag')
#         self.tree.walkTree()
        outputData = self.tree.walkTree(attribute='psi', storeOutput = True)
        self.assertEqual(np.shape(outputData), (self.tree.treeSize, 4), "Walk output array not expected shape.")
        x = outputData[:,0]
        y = outputData[:,1]
        z = outputData[:,2]
        psi = outputData[:,3]
        self.assertIsInstance(psi, np.ndarray, "output array not a numpy array")
        self.assertEqual(np.shape(y), np.shape(psi), "midpoint y values not same shape as psi")
        self.assertEqual(x[0], (self.xmax+self.xmin)/2, "first x value not the center of domain")
        self.assertEqual(z[1], (self.zmax+3*self.zmin)/4, "second z value not the center of domain lower z octant")
        self.assertEqual(psi[44], trueWavefunction(1, x[44], y[44], z[44]), "output psi value doesn't match analytic")
    
    def testPotentialComputation(self):
        self.tree.computePotentialOnTree(epsilon=0)
        print('\nComputed potential: ', self.tree.totalPotential)
        
    def testKineticComputation(self):
        self.tree.computeKineticOnTree()
        print('\nComputed kinetic:    ', self.tree.totalKinetic)
            
if __name__ == "__main__":
    unittest.main()