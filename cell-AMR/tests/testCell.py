'''
Created on Feb 4, 2018

@author: nathanvaughn
'''
import unittest
import numpy as np

from cell_data_structure import cell

class TestCell(unittest.TestCase):


    def setUp(self):
        ''' Set up a 3x3x3 cell on [0,1]x[0,1]x[0,1].  Define psi = exp(-r) '''
        x = y = z = np.array([0.0,0.5,1.0])
        psi = np.zeros((3,3,3))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    r = np.sqrt((i*i + j*j + k*k)/4)
                    psi[i,j,k] = np.exp(-r)
        self.Cell = cell(x,y,z,psi)

    def testPsiEvaluation(self):
        self.assertEqual(self.Cell.psi[0,0,0], np.exp(0), "psi not as expected at (0,0,0)")
        self.assertEqual(self.Cell.psi[2,2,2], np.exp(-np.sqrt(3)), "psi not as expected at (1,1,1)")
        self.assertEqual(self.Cell.psi[1,1,1], np.exp(-np.sqrt(3/4)), "psi not as expected at (0.5,0.5,0.5)")
        self.assertEqual(self.Cell.psi[0,1,1], np.exp(-np.sqrt(2/4)), "psi not as expected at (0,0.5,0.5)")
        
    def testDxDyDz(self):
        self.assertEqual(self.Cell.dx, 0.5, "dx not correct")
        self.assertEqual(self.Cell.dy, 0.5, "dy not correct")
        self.assertEqual(self.Cell.dz, 0.5, "dz not correct")

    def testGradientPsi(self):
        self.Cell.gradient_psi()
        self.assertEqual(np.shape(self.Cell.grad), (3,3,3,3), "gradient not expected shape")
        self.assertEqual(self.Cell.grad[0][1,0,0], (self.Cell.psi[2,0,0]-self.Cell.psi[0,0,0])/(2*self.Cell.dx), "x gradient not as expected along x axis")
        self.assertEqual(self.Cell.grad[2][0,0,1], (self.Cell.psi[0,0,2]-self.Cell.psi[0,0,0])/(2*self.Cell.dz), "z gradient not as expected along z axis")
        


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()