'''
Created on Feb 17, 2018

@author: nathanvaughn
'''
import unittest
from src.mesh import Mesh


class Test(unittest.TestCase):


    def setUp(self):
        self.xmin = self.ymin = 0
        self.xmax = self.ymax = 1
        self.nx = self.ny = 2
        self.mesh = Mesh(self.xmin,self.xmax,self.nx,self.ymin,self.ymax,self.ny)


    def tearDown(self):
        self.mesh = None


    def testInitialMesh(self):
        self.assertEqual(self.mesh.cells[0,0].gridpoints[0,0].x, self.xmin)
        self.assertEqual(self.mesh.cells[0,0].gridpoints[1,1].x, self.xmin + (self.xmax-self.xmin)/(2*self.nx))
        self.assertEqual(self.mesh.cells[0,0].gridpoints[1,2].y, self.ymin + (self.ymax-self.ymin)/(self.ny))
        self.assertEqual(self.mesh.cells[1,1].gridpoints[2,2].x, self.xmax)
        self.assertEqual(self.mesh.cells[1,1].gridpoints[1,2].y, self.ymax)
        
    def testChangingGridpointPropogation(self):
        self.assertEqual(self.mesh.cells[0,0].gridpoints[0,2].x, self.mesh.cells[0,1].gridpoints[0,0].x, "Not same gridpoint")
        self.assertEqual(self.mesh.cells[0,0].gridpoints[0,2].y, self.mesh.cells[0,1].gridpoints[0,0].y, "Not same gridpoint")
        self.assertEqual(self.mesh.cells[0,0].gridpoints[0,2], self.mesh.cells[0,1].gridpoints[0,0], 
                         "00 Cell's top left gridpoint should be the top left cell's bottom left gridpoint")
        
        # check gridpoint value in cell 01.  Then modify gridpoint value in cell 00.  Then check gridpoint value in 01 again.
        print('cell01, gridpoint00 initial x value: ', self.mesh.cells[0,1].gridpoints[0,0].x)
        print('cell00, gridpoint02 initial x value: ', self.mesh.cells[0,0].gridpoints[0,2].x)
        self.assertEqual(self.mesh.cells[0,1].gridpoints[0,0].x, self.xmin, 
                         "Initial x coord of this gridpoint wasn't 0 as expected")
        
        self.mesh.cells[0,0].gridpoints[0,2].x = -55
        print('cell01, gridpoint00 modified x value: ', self.mesh.cells[0,1].gridpoints[0,0].x)
        print('cell00, gridpoint02 modified x value: ', self.mesh.cells[0,0].gridpoints[0,2].x)
        self.assertEqual(self.mesh.cells[0,1].gridpoints[0,0].x, -55, 
                         "Modified x coord of this gridpoint wasn't -55 as expected")


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()