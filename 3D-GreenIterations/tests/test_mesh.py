'''
Created on Jan 21, 2018

@author: nathanvaughn
'''
import unittest
from mesh3D import generate_grid, normalize_wavefunction, np

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
        psi = np.random.rand(10,10,10)
        dx = dy = dz = 0.1
        psi = normalize_wavefunction(psi, dx, dy, dz)
        self.assertAlmostEqual(np.sum(psi*psi)*dx*dy*dz, 1,13, "random wavefunction not normalized properly.")
        

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()