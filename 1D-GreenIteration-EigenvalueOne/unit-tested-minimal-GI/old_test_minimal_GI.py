<<<<<<< HEAD
'''
Created on Jan 15, 2018

@author: nathanvaughn
'''
import unittest
from models import *
from minimal_green_iterations import *
from refactored_functions import *

class TestGreenIterations(unittest.TestCase):


    def setUp(self):
        self.run = setup_Poschl_Teller()

    def tearDown(self):
        pass

    def testGroundStateEnergy(self):
        self.assertEqual(-8.0,self.run.true_energy(self.run.N,4))
        
    def testGrid(self):
        self.assertEqual(self.run.xmin, self.run.grid[0])
        self.assertEqual(self.run.xmax, self.run.grid[-1])
        self.assertEqual(len(self.run.grid), self.run.nx)
        
    def testHamiltonianGenerator(self):
        self.run.add(operanda=2,operandb=3)
        pass


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
=======
'''
Created on Jan 15, 2018

@author: nathanvaughn
'''
import unittest
from models import *
from minimal_green_iterations import *
from refactored_functions import *

class TestGreenIterations(unittest.TestCase):


    def setUp(self):
        self.run = setup_Poschl_Teller()

    def tearDown(self):
        pass

    def testGroundStateEnergy(self):
        self.assertEqual(-8.0,self.run.true_energy(self.run.N,4))
        
    def testGrid(self):
        self.assertEqual(self.run.xmin, self.run.grid[0])
        self.assertEqual(self.run.xmax, self.run.grid[-1])
        self.assertEqual(len(self.run.grid), self.run.nx)
        
    def testHamiltonianGenerator(self):
        self.run.add(operanda=2,operandb=3)
        pass


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
>>>>>>> refs/remotes/eclipse_auto/master
    unittest.main()