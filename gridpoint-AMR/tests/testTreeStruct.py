import unittest
import numpy as np
import io
from contextlib import redirect_stdout


from dataStructs import Tree, Cell, GridPoint
from hydrogenPotential import trueWavefunction
# from timer import Timer

class TestTreeStructure(unittest.TestCase):
    '''
    Series of tests that probe the tree structure.  They verify the relationships between 
    cells, such as parent/child, as well as the fact that neighboring cells have the same
    gridpoint object, not duplicates.  Verifies that one cell modifying gridpoints is felt
    by any other cell that also owns that gridpoint.  Verifies that the neighbor lists are
    generated correctly for a few hand-determined cases.  Lastly, performs the treeWalk, 
    which can be manually inspected to verify the walk does what it is expected.
    '''

    @classmethod
    def setUpClass(self):
        '''
        Generate a mesh on [-1,1]x[-1,1]x[-1,1] containing 8 total cells for testing.  
        setUp() gets called before every test below.
        '''
        print("Default tree used for all tests except where a new test-specific tree is built")
        self.xmin = self.ymin = self.zmin = -12
        self.xmax = self.ymax = self.zmax = 12
        self.tree = Tree(self.xmin,self.xmax,self.ymin,self.ymax,self.zmin,self.zmax)
        self.tree.buildTree( minLevels=0, maxLevels=3, divideTolerance=0.0125, printTreeProperties=True)
        
#     @unittest.skip("Cousins don't share gridpoints yet")
    def testNeighborsPointToSameObject(self):
        '''
        Test 1:  Assert that different cells point to the same grid point object in memory.
        '''
        # check that the bottom left cell and top left cell both point to same gridpoint at their boundary
        self.assertEqual(self.tree.root.children[0,0,1].gridpoints[0,2,1], self.tree.root.children[0,1,1].gridpoints[0,0,1], 
                         "Failed Test: 00 Cell's top left gridpoint object should be the top left cell's bottom left gridpoint object")
        
        # test that cousins share the same gridpoints 
        cousin1 = self.tree.root.children[0,0,0].children[1,1,1]
        cousin2 = self.tree.root.children[0,0,1].children[1,1,0]
        self.assertEqual(cousin2.uniqueID, [element[1] for element in cousin1.neighbors if element[0] == 'zHigh'][0], "cousin 2 isn't zHigh neighbor of cousin 1")
        # cousin1 and cousin2 have different parents but share a face.  
        # the top face of cousin 1 is the bottom face of cousin 2
        self.assertEqual(cousin1.gridpoints[1,1,2], cousin2.gridpoints[1,1,0], "cousins are not sharing same gridpoint as they should.")
        
        cousin3 = self.tree.root.children[0,0,0].children[1,1,1]
        cousin4 = self.tree.root.children[1,0,0].children[0,1,1]
#         print('cousin 3: \n',cousin3.gridpoints)
#         print('cousin 4: \n',cousin4.gridpoints)
        self.assertEqual(np.array_equal(cousin3.gridpoints[2,:,:], cousin4.gridpoints[0,:,:]),True, "cousins 3 and 4 don't share a whole face")
        
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
        
    
        
    def testNeighborLists(self):
        '''
        Test the neighbor list generator.  Built a full tree of depth 3.  Find neighbors of the leaves.
        The third case forces the neighbor finder to go back up to the root.
        '''
        self.tree = Tree(self.xmin,self.xmax,self.ymin,self.ymax,self.zmin,self.zmax)
        self.tree.buildTree( minLevels=3, maxLevels=3, divideTolerance=0.0, printTreeProperties=False)
#         print('length of master list: ', len(self.tree.masterList))
        targetID = '111222111'
        expectedNeighbors = [['zHigh',    '111222112'], 
                             ['zLow',     '111221112'],
                             ['yHigh',  '111222121'],
                             ['yLow',   '111212121'],
                             ['xHigh',    '111222211'],
                             ['xLow', '111122211']]

        for element in self.tree.masterList:
            if element[0] == targetID:
                targetCell = element[1]
        
#         print('\ntarget Cell: ',targetCell.uniqueID)
#         print('neighbors = ',targetCell.neighbors)
        for neighbor in expectedNeighbors:
            self.assertTrue(neighbor in targetCell.neighbors, "expected neighbor %s not in list." %neighbor)
#             print('finding neighbors for first test case')
        targetID = '121122211' 
        targetCell = None
        
#         expectedNeighbors = ['121122212', '121121212', '121122221', '121112221', '121222111', '121122111']
        expectedNeighbors = [['zHigh',    '121122212'], 
                             ['zLow',     '121121212'],
                             ['yHigh',  '121122221'],
                             ['yLow',   '121112221'],
                             ['xHigh',    '121222111'],
                             ['xLow', '121122111']]
        for element in self.tree.masterList:
#             print(element)
            if element[0] == targetID:
                targetCell = element[1]
        
#         print('\ntarget Cell: ',targetCell.uniqueID)
#         print('neighbors = ',targetCell.neighbors)
        for neighbor in expectedNeighbors:
            self.assertTrue(neighbor in targetCell.neighbors, "expected neighbor %s not in list." %neighbor)
        
        targetID = '111122212'  # test case designed so that the neighbor finder must go up to root to find top neighbor
#         expectedNeighbors = ['112121211', '111122211', '111122222', '111112222', '111222112', '111122112']
        expectedNeighbors = [['zHigh',    '112121211'], 
                             ['zLow',     '111122211'],
                             ['yHigh',  '111122222'],
                             ['yLow',   '111112222'],
                             ['xHigh',    '111222112'],
                             ['xLow', '111122112']]
        for element in self.tree.masterList:
            if element[0] == targetID:
                targetCell = element[1]
#         print('\ntarget Cell: ',targetCell.uniqueID)
#         print('neighbors = ',targetCell.neighbors)
        for neighbor in expectedNeighbors:
            self.assertTrue(neighbor in targetCell.neighbors, "expected neighbor %s not in list." %neighbor)
        
                  
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
        self.assertEqual(parent.gridpoints[2,0,2], child.gridpoints[2,0,2], "parent and child don't have same corner point.")
    
    def testAnalyticPsiSetting(self):
        self.assertEqual(self.tree.root.gridpoints[1,1,1].psi, trueWavefunction(1, 0, 0, 0), "root midpoint (the origin) doesn't have correct wavefunction.")
    

#     @unittest.skip("Skip energy computations.")
    def testPotentialComputation(self):
        self.tree.computePotentialOnTree(epsilon=0)
        print('\nPotential Error:        %.3g mHartree' %float((-1.0-self.tree.totalPotential)*1000.0))
#     @unittest.skip("Skip energy computations.")    
    def testKineticComputation(self):
        self.tree.computeKineticOnTree()
        print('\nKinetic Error:           %.3g mHartree' %float((0.5-self.tree.totalKinetic)*1000.0))
            

    @unittest.skip("skip manual eye-tests")
    def testTreeWalk(self):
#         self.tree.walkTree(attribute='divideFlag')
        self.tree.walkTree(leavesOnly=True)
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
    
    def testNumberOfNewlyGeneratedCells(self):
        '''
        verify that if all neighbors have already divided, 
        only 26 new gridpoints generatred.  Manually divide the 6 neighbors,
        then divide the center cell and output the number of new gridpoints 
        generated.  
        '''
        self.tree = None
        self.tree = Tree(self.xmin,self.xmax,self.ymin,self.ymax,self.zmin,self.zmax)
        self.tree.buildTree( minLevels=2, maxLevels=2, divideTolerance=0.0, printTreeProperties = False)
        testCell = self.tree.root.children[0,0,0].children[1,1,1]
        self.tree.root.children[0,0,0].children[1,1,0].divide()
        self.tree.root.children[0,0,0].children[1,0,1].divide()
        self.tree.root.children[0,0,0].children[0,1,1].divide()
        self.tree.root.children[0,0,1].children[1,1,0].divide()
        self.tree.root.children[0,1,0].children[1,0,1].divide()
        self.tree.root.children[1,0,0].children[0,1,1].divide()
        
        ''' captures the 'prints' from divide() as output, which can then be tested. '''
        f = io.StringIO()
        with redirect_stdout(f):
            testCell.divide(printNumberOfCells=True)
        cellDivisionPrint = f.getvalue()

        self.assertEqual(cellDivisionPrint, "generated 26 new gridpoints for parent cell 111222\n", 
                "Cell did not generate exactly 26 new gridpoints as it should have.")
    
if __name__ == "__main__":
    unittest.main()