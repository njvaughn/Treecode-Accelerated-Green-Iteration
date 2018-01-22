'''
Created on Jan 22, 2018

@author: nathanvaughn
'''
import unittest

from hamiltonian import Hamiltonian, Delta
from mesh3D import generate_grid
from hydrogen_potential import potential

class TestHamiltonian(unittest.TestCase):

    def testSecondDerivative(self):
        '''
        Verify that the second derivative operator performs as expected for linear and 
        quadratic functions in 3D
        '''
        xmin = -1; xmax = 1
        ymin = -1; ymax = 1
        zmin = -1; zmax = 1
        nx = ny = nz = 20+1
        x,y,z = generate_grid(nx, ny, nz, xmin, xmax, ymin, ymax, zmin, zmax)
        psi = x*x + y*y + z*z  # quadratic function, laplacian should return constant?
        Delta_psi = Delta(psi,x,y,z) 
        self.assertAlmostEqual(Delta_psi[11][9][10], 6,12, 
                               "Second derivative not producing correct result for quadratic.")
        self.assertAlmostEqual(Delta_psi[0,1,3], Delta_psi[10,18,0], 10, 
                               "Second derivative isn't constant throughout domain.")
        
        psi = x + y + z  # linear function, laplacian should return zero
        Delta_psi = Delta(psi,x,y,z)
        self.assertAlmostEqual(Delta_psi[11][9][10], 0,12, 
                               "Second derivative not producing correct result for linear.")
        self.assertAlmostEqual(Delta_psi[0,1,3], Delta_psi[10,18,0], 10, 
                               "Second derivative isn't constant throughout domain.")
        
    def testHamiltonian(self):
        xmin = -1; xmax = 1
        ymin = -1; ymax = 1
        zmin = -1; zmax = 1
        nx = ny = nz = 20
        x,y,z = generate_grid(nx, ny, nz, xmin, xmax, ymin, ymax, zmin, zmax)
        psi = x*x + y*y + z*z
        Hamiltonian(potential(x,y,z), psi, x, y, z)
        

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()