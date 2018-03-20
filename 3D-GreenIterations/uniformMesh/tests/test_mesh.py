'''
Created on Jan 21, 2018

@author: nathanvaughn
'''
import unittest
import numpy as np
from mesh3D import generate_grid, normalize_wavefunction, simpson_weight_matrix, trapezoid_weight_matrix

class Test_Mesh(unittest.TestCase):


    def testGrid(self):
        xmin = -1; xmax = 1
        ymin = -2; ymax = 2
        zmin = -3; zmax = 3
        nx = ny = nz = 20+1
        randi = np.random.randint(nx)
        randj = np.random.randint(ny)
        randk = np.random.randint(nz)
        nx = ny = nz = 20+1
        x,y,z = generate_grid(nx, ny, nz, xmin, xmax, ymin, ymax, zmin, zmax)
        self.assertEqual(np.shape(x), (21,21,21), 'grid not expected size')
        self.assertEqual(np.shape(x), np.shape(y), 'x and y grid dimensions not matchings')  
        self.assertEqual(np.shape(x), np.shape(z), 'x and z grid dimensions not matchings')  
        self.assertEqual(x[0,randj,randk], xmin, "grid not storing xmin where expected")
        self.assertEqual(x[-1,randj,randk], xmax, "grid not storing xmax where expected")
        self.assertEqual(y[randi,0,randk], ymin, "grid not storing ymin where expected")
        self.assertEqual(y[randi,-1,randk], ymax, "grid not storing ymax where expected")
        self.assertEqual(z[randi,randj,0], zmin, "grid not storing zmin where expected")
        self.assertEqual(z[randi,randj,-1], zmax, "grid not storing zmax where expected")
        
    def testSmallGrid(self):
        '''
        Verify how the grid is being set up.  min and max values are the endpoints (includively).
        '''
        nx = ny = nz = 2
        xmin = ymin = zmin = -1
        xmax = ymax = zmax = 1
        self.assertEqual(np.linspace(xmin,xmax,nx)[0], -1 , "small grid not performing as expected")
        self.assertEqual(np.linspace(xmin,xmax,nx)[1], 1 , "small grid not performing as expected")
        x,y,z = generate_grid(nx, ny, nz, xmin, xmax, ymin, ymax, zmin, zmax)
        self.assertEqual(x[0,0,0], xmin, "xmin isn't the 000 component of x as expected")
        self.assertEqual(y[0,0,0], ymin, "xmin isn't the 000 component of y as expected")
        self.assertEqual(z[0,0,-1], zmax, "zmax isn't the 00-1 component of z as expected")
        
        
    def testNormalization(self):
        nx = ny = nz = 41
        psi = np.random.rand(nx,ny,nz)
        dx = dy = dz = 0.1
        Wt = trapezoid_weight_matrix(nx,ny,nz)
        psi = normalize_wavefunction(psi, dx, dy, dz,Wt)
        self.assertAlmostEqual(np.sum(Wt*psi*psi)*dx*dy*dz, 1,13, "random wavefunction not normalized properly.")
        

    def testIntegrationSchemes(self):
        nx = ny = nz = 41
        xmin = ymin = zmin = -1
        xmax = ymax = zmax = 1
        dx = dy = dz = (xmax-xmin)/(nx-1)
        x,y,z = generate_grid(nx, ny, nz, xmin, xmax, ymin, ymax, zmin, zmax)
        Ws = simpson_weight_matrix(nx, ny, nz)
        Wt = trapezoid_weight_matrix(nx, ny, nz)
        
        # Integrate a constant function exactly
        f = np.ones((nx,ny,nz))
        I1s = np.sum(Ws*f)*dx*dy*dz
        I1t = np.sum(Wt*f)*dx*dy*dz
        self.assertAlmostEqual(I1s, I1t, 12, "trapezoid and simpson didn/'t agree for constant function")
        self.assertAlmostEqual(I1s, 8, 12, "Integrating constant didn/'t give expected result")
        # Integrate a linear function exactly
        g = x+y+z
        I2s = np.sum(Ws*g)*dx*dy*dz
        I2t = np.sum(Wt*g)*dx*dy*dz
        self.assertAlmostEqual(I2s, I2t, 12, "trapezoid and simpson didn/'t agree for linear function")
        self.assertAlmostEqual(I2s, 0, 12, "Integrating linear didn/'t give expected result")

        # Integrate a quadratic function exactly
        h = x*x+y*y+z*z
        I3s = np.sum(Ws*h)*dx*dy*dz
        I3t = np.sum(Wt*h)*dx*dy*dz
        self.assertNotAlmostEqual(I3s, I3t, 12, "trapezoid and simpson shouldn/'t agree for quadratic function, trap should have error")
        self.assertAlmostEqual(I3s, 8, 12, "Simpson didn/t integrate quadratic correctly")
        
        # Integrate a smooth function.  Check for convergence
        i = x**3 - y**5 + z**4
        I3s = np.sum(Ws*i)*dx*dy*dz
        I3t = np.sum(Wt*i)*dx*dy*dz
        coarse_error = 1.6 - I3s
        coarse_error_m = 1.6 - I3t
        
        nx = ny = nz = 2*(nx-1)+1
        dx = dy = dz = (xmax-xmin)/(nx-1)
        x,y,z = generate_grid(nx, ny, nz, xmin, xmax, ymin, ymax, zmin, zmax)
        Ws = simpson_weight_matrix(nx, ny, nz)
        Wt = trapezoid_weight_matrix(nx, ny, nz)
        i = x**3 - y**5 + z**4
        I3s = np.sum(Ws*i)*dx*dy*dz
        I3t = np.sum(Wt*i)*dx*dy*dz
        fine_error = 1.6 - I3s
        fine_error_m = 1.6 - I3t
        
#         print('coarse error: ', coarse_error)
#         print('fine error:   ', fine_error)
        self.assertAlmostEqual(I3s, 1.6, 4, "Simpson didn/t integrate quadratic correctly")
        self.assertAlmostEqual(coarse_error/fine_error, 16, 4, "Didn/'t observe 4th order convergence for smooth function using Simpson's rule")
        self.assertAlmostEqual(coarse_error_m/fine_error_m, 4, 2, "Didn/'t observe 2nd order convergence for smooth function using Trapezoid rule")
        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()