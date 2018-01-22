'''
Created on Jan 22, 2018

@author: nathanvaughn
'''
import unittest
import numpy as np

from mesh3D import generate_grid
from convolution import conv, Green
from hydrogen_potential import potential


class TestConvolution(unittest.TestCase):


    def testSimpleConvolution(self):
        '''
        8 points on the corners of a [-1,1]x[-1,1]x[-1,1] box.
        Potential is equal at all corners for a hydrogen nucleus at origin
        Wavefunction is constant, psi = 1
        Compute the interaction under the Green's function for Helmholtz (Delta + k^2). 
        Resulting values should be equal at all corners, and should be equal to the \
        manually computed value.
        '''
        x,y,z = generate_grid(2,2,2,-1,1,-1,1,-1,1) # 8 points on the corners of the box [-1,1]^3
        P = potential(x,y,z)
        k=1 # pick any k >= 0.  A different Green's function is required for k<0, more suited for Plane Wave expansions
        psi = np.ones(np.shape(P))
        testConv = conv(P,k,psi,x,y,z)
        self.assertAlmostEqual(testConv[0,0,0], testConv[1,0,1],10, 
                    "convolution didn't respect symmetry of test problem")
        self.assertAlmostEqual(testConv[0,1,0], testConv[1,0,1],10, 
                    "convolution didn't respect symmetry of test problem")
        self.assertAlmostEqual(testConv[1,1,1], testConv[1,0,1],10, 
                    "convolution didn't respect symmetry of test problem")
        r0 = np.sqrt(3)     # V(r) for each of the corners for hydrogen at origin
        r1 = 2              # 3 neighbors distance 2
        r2 = 2*np.sqrt(2)   # 3 nieghbors distance 2sqrt(2)
        r3 = 2*np.sqrt(3)   # 1 neighbot distance 2sqrt(3)
        expectedValue = 1/4/np.pi/r0 * (3*np.exp(-k*r1)/r1 + 3*np.exp(-k*r2)/r2 + np.exp(-k*r3)/r3)
        self.assertAlmostEqual(testConv[0,0,0], expectedValue,10, 
                    "convolution didn't produce expected value for the checked grid point")
        
    def testGreenFunction(self):
        x = 5; y = -1; z = 2; k = 0;
        self.assertEqual(Green(x,y,z,k), -np.exp(k*np.sqrt(x*x+y*y+z*z))/(4*np.pi*np.sqrt(x*x+y*y+z*z)), 
                         "Green Function not giving expected result.")
        self.assertEqual(Green(x, y, z, -1), "ValueError: this Green function is valid for k>=0.", 
                         "Didn't catch k<0 correctly.")


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()