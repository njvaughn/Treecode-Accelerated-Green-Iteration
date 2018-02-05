'''
Created on Feb 4, 2018

@author: nathanvaughn
'''
import unittest
import numpy as np

from cell_data_structure import cell

class TestCell(unittest.TestCase):


    def setUp(self):
        gridpoints = np.array([ [0,0.5,1],[0,0.5,1],[0,0.5,1]])
        radii = 
        psi = np.exp(gridpoints)
        self.Cell = cell(gridpoints,psi)
        


    def tearDown(self):
        pass


    def testPsiEvaluation(self):
        self.assertEqual(self.cell.psi[0,0,0], np.exp(0), msg)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()