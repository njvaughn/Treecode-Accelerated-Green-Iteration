import unittest
from DataStructs import Mesh


class TestDataStructures(unittest.TestCase):


    def setUp(self):
        '''
        Generate a mesh on [0,1]x[0,1] containing 4 total cells for testing.  
        setUp() gets called before every test below.
        '''
        self.xmin = self.ymin = 0
        self.xmax = self.ymax = 1
        self.nx = self.ny = 2
        self.mesh = Mesh(self.xmin,self.xmax,self.nx,self.ymin,self.ymax,self.ny)


    def testNeighborsPointToSameObject(self):
        '''
        Test 1:  Assert that different cells point to the same grid point object in memory.
        '''
        # check that the bottom left cell and top left cell both point to same gridpoint at their boundary
        self.assertEqual(self.mesh.cells[0,0].gridpoints[0,2], self.mesh.cells[0,1].gridpoints[0,0], 
                         "Failed Test: 00 Cell's top left gridpoint object should be the top left cell's bottom left gridpoint object")
        
    def testNeighborsFeelModifications(self):
        '''
        Test 2:  Assert that modifying a certain gridpoint in one cell is felt by all cells that are composed of that gridpoint
        '''        
        # modify a boundary gridpoint's data in bottom left cell to x=-52
        self.mesh.cells[0,0].gridpoints[0,2].x = -52
        # check that the top left cell's gridpoint data now has the updated value.
        self.assertEqual(self.mesh.cells[0,1].gridpoints[0,0].x, -52, 
                         "Failed Test:  Modified x coord of this gridpoint wasn't -52 as expected")


if __name__ == "__main__":
    unittest.main()