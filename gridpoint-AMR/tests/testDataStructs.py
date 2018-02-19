import unittest
from dataStructs import Mesh


class TestDataStructures(unittest.TestCase):


    def setUp(self):
        '''
        Generate a mesh on [0,1]x[0,1] containing 4 total cells for testing.  
        setUp() gets called before every test below.
        '''
        self.xmin = self.ymin = self.zmin = 0
        self.xmax = self.ymax = self.zmax = 1
        self.nx = self.ny = self.nz = 2
        self.mesh = Mesh(self.xmin,self.xmax,self.nx,self.ymin,self.ymax,self.ny,self.zmin,self.zmax,self.nz)


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
        


if __name__ == "__main__":
    unittest.main()