<<<<<<< HEAD
'''
Created on Jan 21, 2018

@author: nathanvaughn
'''
# import tools
import unittest
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import cm

# import methods for testing
from hydrogen_potential import potential, trueEnergy, trueWavefunction
from mesh3D import generate_grid, np


class Test_Hydrogen(unittest.TestCase):


    def testPotential(self):
        x = np.random.rand()
        y = np.random.rand()
        z = np.random.rand()
        v = potential(x,y,z)
        self.assertEqual(v,-1/np.sqrt(x**2 + y**2 + z**2))
        
    def testPotentialOnMesh(self):
        x,y,z = generate_grid(20, 20, 20, -4, 4, -4, 4, -4, 4)
        P = potential(x,y,z)
        self.assertEqual(np.shape(P), np.shape(x), "Potential doesn't have same dimensional as grid")
        self.assertEqual(P[3,4,5], potential(x[3,4,5],y[3,4,5],z[3,4,5]),"Potential at given point doesn't agree with V(that point)")
        self.assertEqual(np.max(P),potential(4,4,4),"max P value not agreeing with V(xmax,ymax,zmax)")


    def test2DPotentialProjection(self):
        nx = ny = nz = 10
        x,y,z = generate_grid(nx, ny, nz, -4, 4, -4, 4, -4, 4)
        P = potential(x,y) # z=0 by default
        self.assertEqual(np.shape(P), (nx,ny,nz), "Projection onto z=0 shape not expected.")
        randk1 = np.random.randint(nz)
        randk2 = np.random.randint(nz)
        self.assertEqual(P[3,4,randk1], P[3,4,randk2], "Each z-slice should be identical.")
    
        ''' Plot the potential on a relatively coarse grid for manual inspection.'''
#         fig = plt.figure()
#         ax = fig.gca(projection='3d')
#         surf = ax.plot_surface(x[:,:,0], y[:,:,0], P[:,:,0], cmap=cm.coolwarm,
#                        linewidth=0, antialiased=True)
#         plt.show()
        
    def testAnalyticEnergy(self):
        pass
    
    def testAnalyticWavefunction(self):
        pass
#         x,y,z = generate_grid(21, 21, 21, -10, 10, -10, 10, -10, 10)
#         dx = dy = dz = 1
#         psi = trueWavefunction(0, x, y, z)
#         
#         self.assertEqual(np.sum(psi*psi)*dx*dy*dz, 1, "wavefunction isn't normalized")
        

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
=======
'''
Created on Jan 21, 2018

@author: nathanvaughn
'''
# import tools
import unittest
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import cm

# import methods for testing
from hydrogen_potential import potential, trueEnergy, trueWavefunction
from mesh3D import generate_grid, np


class Test_Hydrogen(unittest.TestCase):


    def testPotential(self):
        x = np.random.rand()
        y = np.random.rand()
        z = np.random.rand()
        v = potential(x,y,z)
        self.assertEqual(v,-1/np.sqrt(x**2 + y**2 + z**2))
        
    def testPotentialOnMesh(self):
        x,y,z = generate_grid(20, 20, 20, -4, 4, -4, 4, -4, 4)
        P = potential(x,y,z)
        self.assertEqual(np.shape(P), np.shape(x), "Potential doesn't have same dimensional as grid")
        self.assertEqual(P[3,4,5], potential(x[3,4,5],y[3,4,5],z[3,4,5]),"Potential at given point doesn't agree with V(that point)")
        self.assertEqual(np.max(P),potential(4,4,4),"max P value not agreeing with V(xmax,ymax,zmax)")


    def test2DPotentialProjection(self):
        nx = ny = nz = 10
        x,y,z = generate_grid(nx, ny, nz, -4, 4, -4, 4, -4, 4)
        P = potential(x,y) # z=0 by default
        self.assertEqual(np.shape(P), (nx,ny,nz), "Projection onto z=0 shape not expected.")
        randk1 = np.random.randint(nz)
        randk2 = np.random.randint(nz)
        self.assertEqual(P[3,4,randk1], P[3,4,randk2], "Each z-slice should be identical.")
    
        ''' Plot the potential on a relatively coarse grid for manual inspection.'''
#         fig = plt.figure()
#         ax = fig.gca(projection='3d')
#         surf = ax.plot_surface(x[:,:,0], y[:,:,0], P[:,:,0], cmap=cm.coolwarm,
#                        linewidth=0, antialiased=True)
#         plt.show()
        
    def testAnalyticEnergy(self):
        pass
    
    def testAnalyticWavefunction(self):
        pass
#         x,y,z = generate_grid(21, 21, 21, -10, 10, -10, 10, -10, 10)
#         dx = dy = dz = 1
#         psi = trueWavefunction(0, x, y, z)
#         
#         self.assertEqual(np.sum(psi*psi)*dx*dy*dz, 1, "wavefunction isn't normalized")
        

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
>>>>>>> refs/remotes/eclipse_auto/master
    unittest.main()