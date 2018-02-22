import unittest
import numpy as np


from dataStructs import Mesh, Tree, Cell, GridPoint
from hydrogenPotential import potential, trueWavefunction

class TestDataStructures(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        '''
        Generate a mesh on [-1,1]x[-1,1]x[-1,1] containing 8 total cells for testing.  
        setUp() gets called before every test below.
        '''
        self.xmin = self.ymin = self.zmin = -5
        self.xmax = self.ymax = self.zmax = 5
        self.nx = self.ny = self.nz = 2
        self.mesh = Mesh(self.xmin,self.xmax,self.nx,self.ymin,self.ymax,self.ny,self.zmin,self.zmax,self.nz)
        self.tree = Tree(self.xmin,self.xmax,self.ymin,self.ymax,self.zmin,self.zmax)
        self.tree.buildTree( minLevels=0, maxLevels=5, divideTolerance=0.0025)

    def testNeighborsPointToSameObject(self):
        '''
        Test 1:  Assert that different cells point to the same grid point object in memory.
        '''
        # check that the bottom left cell and top left cell both point to same gridpoint at their boundary
        self.assertEqual(self.mesh.cells[0,0,1].gridpoints[0,2,1], self.mesh.cells[0,1,1].gridpoints[0,0,1], 
                         "Failed Test: 00 Cell's top left gridpoint object should be the top left cell's bottom left gridpoint object")
        
    def testNeighborsFeelModifications(self):
        '''
        Test 2:  Assert that modifying a certain gridpoint in one cell is felt by all cells that are composed of that gridpoint
        '''        
        # modify a boundary gridpoint's data in bottom left cell to x=-52
        self.mesh.cells[0,0,0].gridpoints[0,2,0].x = -52
        # check that the top left cell's gridpoint data now has the updated value.
        self.assertEqual(self.mesh.cells[0,1,0].gridpoints[0,0,0].x, -52, 
                         "Failed Test:  Modified x coord of this gridpoint wasn't -52 as expected")
        
        self.mesh.cells[0,0,0].gridpoints[0,2,0].setPsi(0.55)
        self.assertEqual(self.mesh.cells[0,0,0].gridpoints[0,2,0].psi, 0.55, 
                         "Failed Test:  Psi not set properly")
        self.assertEqual(self.mesh.cells[0,1,0].gridpoints[0,0,0].psi, 0.55, 
                         "Failed Test:  Psi set from neighboring cell not noticed")
        
    def testCellDivide(self):
        '''
        Test 3:  Test cell division.  The original 3x3x3 GridPoint objects should be pointed to by the children
        in addition to the new GridPoint objects created at the refined level.  Check that the gridpoint data 
        gets mapped properly (children are in fact the 8 octants).
        '''
        self.mesh.cells[0,0,0].divide()
        parent = self.mesh.cells[0,0,0]
        # check that previously existing object are now also owned by the children
        self.assertEqual(parent.gridpoints[2,2,2], parent.children[1,1,1].gridpoints[2,2,2],
                          "corner point not mapped to expected child")
        self.assertEqual(parent.gridpoints[1,1,1], parent.children[1,1,1].gridpoints[0,0,0],
                          "middle point not mapped to expected child")
        self.assertEqual(parent.gridpoints[2,0,1], parent.children[1,0,1].gridpoints[2,0,0],
                          "corner point not mapped to expected child")
        
        # check that children's new cells have the correct new gridpoints
        self.assertEqual(parent.children[0,0,0].gridpoints[1,1,1].x, 
                         (parent.gridpoints[0,0,0].x + parent.gridpoints[1,0,0].x )/2, 
                         "midpoint of child cell not expected value.")
        
        # check that children own same objects on their boundaries
        self.assertEqual(parent.children[0,0,0].gridpoints[1,1,2], parent.children[0,0,1].gridpoints[1,1,0], 
                         "Neighboring children aren't pointing to same gridpoint object on their shared face")
        
    def testObjectTypesAfterTreeConstruction(self):
        self.assertIsInstance(self.tree, Tree, "self is not a Tree object.")
        self.assertIsInstance(self.tree.root, Cell, "Root of tree is not a cell object.")
        self.assertIsInstance(self.tree.root.gridpoints[0,0,2], GridPoint, "Gridpoints of root are not GridPoint objects.")
        self.assertIsInstance(self.tree.root.children[0,1,0], Cell, "Children of root are not cell objects.")
        self.assertIsInstance(self.tree.root.children[0,1,0].gridpoints[2,1,2], GridPoint, "Children's gridpoints are not GridPoint objects.")
        self.assertIsInstance(self.tree.root.children[0,1,0].gridpoints[2,1,2].x, np.float, "Gridpoint data not numpy floats.")
        
    def testTreeBuild(self):
        self.assertEqual(self.tree.root.level, 0, "Root level wasn't equal to 0.")
        self.assertEqual(self.tree.root.gridpoints[0,0,0].x, self.xmin, "Root's 000 corner point doesn't have correct x value")
        self.assertEqual(self.tree.root.children[1,0,1].level, 1, "Root's children level wasn't equal to 1.")
#         self.assertEqual(self.tree.maxDepth, 3, "depth wasn't 2 like expected.  This test depends on domain size and division rule.  Not very general.")
       
    def testTreePointers(self):
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
        self.assertEqual(psi[44], trueWavefunction(1, x[44], y[44], z[44]), "psi value doesn't match analytic")
            
if __name__ == "__main__":
    unittest.main()