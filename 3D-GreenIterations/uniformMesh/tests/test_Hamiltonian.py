<<<<<<< HEAD
'''
Created on Jan 22, 2018

@author: nathanvaughn
'''
import unittest
import numpy as np

from hamiltonian import Hamiltonian, Delta, EnergyUpdate
from mesh3D import generate_grid, normalize_wavefunction, trapezoid_weight_matrix, simpson_weight_matrix
from hydrogen_potential import potential, potential_smoothed, trueWavefunction, trueEnergy

class TestHamiltonian(unittest.TestCase):

    def testSecondDerivative(self):
        '''
        Verify that the second derivative operator performs as expected for linear and 
        quadratic functions in 3D
        '''
        xmin = -1; xmax = 1
        ymin = -1; ymax = 1
        zmin = -1; zmax = 1
        nx = ny = nz = 20
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
        


#     def testHydrogenGroundStateEnergy(self):
#         xmin = -10; xmax = 10
#         ymin = zmin = xmin
#         ymax = zmax = xmax
#         nx = ny = nz = 21
#         dx = dy = dz = (xmax-xmin)/(nx-1)
#         x,y,z = generate_grid(nx, ny, nz, xmin, xmax, ymin, ymax, zmin, zmax)
#         W = trapezoid_weight_matrix(nx, ny, nz)
#         self.assertAlmostEqual(dx, x[1,0,0]-x[0,0,0],10, "dx not as expected.  21 grid points spanning [-10,10] should have dx=1")
#         psi = trueWavefunction(1,x,y,z)
#         psi = normalize_wavefunction(psi, dx, dy, dz, W)
#         V = potential_smoothed(x,y,z,0.1)
#         computed_energy = EnergyUpdate(V, psi, x, y, z, W)
#         true_energy = trueEnergy(1)
#         self.assertAlmostEqual(computed_energy, true_energy,2, "Hamiltonian not giving correct energy from analytic psi") 

    def testPotentialEnergy(self):
        xmin = -8; xmax = 8.25
        ymin = zmin = xmin
        ymax = zmax = xmax
        nx = ny = nz = 81
        dx = dy = dz = (xmax-xmin)/(nx-1)
#         W = trapezoid_weight_matrix(nx, ny, nz)
        W = simpson_weight_matrix(nx, ny, nz)
        x,y,z = generate_grid(nx, ny, nz, xmin, xmax, ymin, ymax, zmin, zmax)
        self.assertAlmostEqual(dx, x[1,0,0]-x[0,0,0],10, "dx not as expected.")
        psi = trueWavefunction(1,x,y,z)
        psi = normalize_wavefunction(psi, dx, dy, dz, W)
        V = potential(x, y, z)
#         V = potential_smoothed(x, y, z, dx/2)
        computed_PE = np.sum(W*psi*V*psi)*dx*dy*dz
        true_PE = -1.0
        potential_error = true_PE - computed_PE
        print('PE error: ', potential_error)
        self.assertAlmostEqual(computed_PE, true_PE, 2, "potential energy not accurate enough")
    
    def testKineticEnergy(self):
        xmin = -8; xmax = 8
        ymin = zmin = xmin
        ymax = zmax = xmax
        nx = ny = nz = 41
        dx = dy = dz = (xmax-xmin)/(nx-1)
        W = trapezoid_weight_matrix(nx, ny, nz)
        x,y,z = generate_grid(nx, ny, nz, xmin, xmax, ymin, ymax, zmin, zmax)
        self.assertAlmostEqual(dx, x[1,0,0]-x[0,0,0],10, "dx not as expected.")
        psi = trueWavefunction(1,x,y,z)
        psi = normalize_wavefunction(psi, dx, dy, dz, W)
        delta_psi = Delta(psi, x, y, z)
        computed_kinetic = -1/2*np.sum(W*psi*delta_psi)*dx*dy*dz
        true_kinetic = 0.5
        kinetic_error = true_kinetic - computed_kinetic
        print('KE error: ', kinetic_error)
        self.assertAlmostEqual(computed_kinetic, true_kinetic, 1, "kinetic not accurate enough")
        
    
    def testRichardsonExtrapolationPE(self):
        xmin = -8; xmax = 8
        ymin = zmin = xmin
        ymax = zmax = xmax
        k=2
        # coarse run
        nxc = nyc = nzc = 41
        W = trapezoid_weight_matrix(nxc, nyc, nzc)
        dxc = dyc = dzc = (xmax-xmin)/(nxc-1)
        x,y,z = generate_grid(nxc, nyc, nzc, xmin, xmax, ymin, ymax, zmin, zmax)
        self.assertAlmostEqual(dxc, x[1,0,0]-x[0,0,0],10, "dx not as expected.")
        psi = trueWavefunction(1,x,y,z)
        psi = normalize_wavefunction(psi, dxc, dyc, dzc, W)
#         V = potential(x, y, z)
        V = potential_smoothed(x, y, z, dxc/2)
        computed_PE_c = np.sum(W*psi*V*psi)*dxc*dyc*dzc
        true_PE = -1.0
        potential_error = true_PE - computed_PE_c
        print('coarse grid PE error: ', potential_error)
        
        # fine run
        nxf = nyf = nzf = k*(nxc-1)+1
        dxf = dyf = dzf = (xmax-xmin)/(nxf-1)
        W = trapezoid_weight_matrix(nxf, nyf, nzf)
        x,y,z = generate_grid(nxf, nyf, nzf, xmin, xmax, ymin, ymax, zmin, zmax)
        self.assertAlmostEqual(dxf, x[1,0,0]-x[0,0,0],10, "dx not as expected.")
        psi = trueWavefunction(1,x,y,z)
        psi = normalize_wavefunction(psi, dxf, dyf, dzf, W)
#         V = potential(x, y, z)
        V = potential_smoothed(x, y, z, dxf/2)
        computed_PE_f = np.sum(W*psi*V*psi)*dxf*dyf*dzf
        true_PE = -1.0
        potential_error = true_PE - computed_PE_f
        print('fine grid PE error: ', potential_error)
        
        # extrpolated value
        extrapolated_PE = (k**2*computed_PE_f - computed_PE_c) / (k**2 - 1)
        extrapolated_error = true_PE - extrapolated_PE
        print('extrapolated PE error: ', extrapolated_error)
#         self.assertEqual(0, 1, "0 not equal to 1")
        
    def testRichardsonExtrapolationKE(self):
        xmin = -8; xmax = 8
        ymin = zmin = xmin
        ymax = zmax = xmax
        k=2
        # coarse run
        nxc = nyc = nzc = 21
        dxc = dyc = dzc = (xmax-xmin)/(nxc-1)
#          Wc = trapezoid_weight_matrix(nxc,nyc,nzc)
        Wc = simpson_weight_matrix(nxc,nyc,nzc)
        x,y,z = generate_grid(nxc, nyc, nzc, xmin, xmax, ymin, ymax, zmin, zmax)
        self.assertAlmostEqual(dxc, x[1,0,0]-x[0,0,0],10, "dx not as expected.")
        psi = trueWavefunction(1,x,y,z)
        psi = normalize_wavefunction(psi, dxc, dyc, dzc, Wc)
        delta_psi = Delta(psi, x, y, z)
        computed_KE_c = -1/2*np.sum(Wc*psi*delta_psi)*dxc*dyc*dzc
        true_KE = 0.5
        kinetic_error = true_KE - computed_KE_c
        print('coarse grid KE error: ', kinetic_error)
        
        # fine run
        nxf = nyf = nzf =  k*(nxc-1)+1  # if nxc odd, use k*(nxc-1)+1.  Else use k*nxc
        dxf = dyf = dzf = (xmax-xmin)/(nxf-1)
#         Wf = trapezoid_weight_matrix(nxf,nyf,nzf)
        Wf = simpson_weight_matrix(nxf,nyf,nzf)
        x,y,z = generate_grid(nxf, nyf, nzf, xmin, xmax, ymin, ymax, zmin, zmax)
        self.assertAlmostEqual(dxf, x[1,0,0]-x[0,0,0],10, "dx not as expected.")
        psi = trueWavefunction(1,x,y,z)
        psi = normalize_wavefunction(psi, dxf, dyf, dzf, Wf)
        delta_psi = Delta(psi, x, y, z)
        computed_KE_f = -1/2*np.sum(Wf*psi*delta_psi)*dxf*dyf*dzf
        true_KE = 0.5
        kinetic_error = true_KE - computed_KE_f
        print('fine grid KE error: ', kinetic_error)
        
        # extrpolated value
        extrapolated_KE = (k**2*computed_KE_f - computed_KE_c) / (k**2 - 1)
        extrapolated_error = true_KE - extrapolated_KE
        print('extrapolated KE error: ', extrapolated_error)
        self.assertEqual(0, 1, "0 not equal to 1")
        
        
        
#     def testGSConvergence(self):
#         xmin = -10; xmax = 10
#         ymin = zmin = xmin
#         ymax = zmax = xmax
#         nx = ny = nz = 20
#         dx = dy = dz = (xmax-xmin)/(nx-1)
#         x,y,z = generate_grid(nx, ny, nz, xmin, xmax, ymin, ymax, zmin, zmax)
#         self.assertAlmostEqual(dx, x[1,0,0]-x[0,0,0],10, "dx not as expected.")
#         psi = trueWavefunction(1,x,y,z)
#         psi = normalize_wavefunction(psi, dx, dy, dz)
#         V = potential_smoothed(x,y,z,0.1)
#         computed_energy_coarse = EnergyUpdate(V, psi, x, y, z)
#         
#         nx = ny = nz = 40
#         dx = dy = dz = (xmax-xmin)/(nx-1)
#         x,y,z = generate_grid(nx, ny, nz, xmin, xmax, ymin, ymax, zmin, zmax)
#         self.assertAlmostEqual(dx, x[1,0,0]-x[0,0,0],10, "dx not as expected.")
#         psi = trueWavefunction(1,x,y,z)
#         psi = normalize_wavefunction(psi, dx, dy, dz)
#         V = potential_smoothed(x,y,z,0.1)
#         computed_energy_fine = EnergyUpdate(V, psi, x, y, z)
#         true_energy = trueEnergy(1)
#         print('coarse error: ', abs(computed_energy_coarse-true_energy))
#         print('fine error:   ', abs(computed_energy_fine-true_energy))
#         self.assertLess(abs(computed_energy_fine-true_energy), abs(computed_energy_coarse-true_energy), "Fine mesh didn't give more accurate energy")
    
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
=======
'''
Created on Jan 22, 2018

@author: nathanvaughn
'''
import unittest
import numpy as np

from hamiltonian import Hamiltonian, Delta, EnergyUpdate
from mesh3D import generate_grid, normalize_wavefunction, trapezoid_weight_matrix, simpson_weight_matrix
from hydrogen_potential import potential, potential_smoothed, trueWavefunction, trueEnergy

class TestHamiltonian(unittest.TestCase):

    def testSecondDerivative(self):
        '''
        Verify that the second derivative operator performs as expected for linear and 
        quadratic functions in 3D
        '''
        xmin = -1; xmax = 1
        ymin = -1; ymax = 1
        zmin = -1; zmax = 1
        nx = ny = nz = 20
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
        


#     def testHydrogenGroundStateEnergy(self):
#         xmin = -10; xmax = 10
#         ymin = zmin = xmin
#         ymax = zmax = xmax
#         nx = ny = nz = 21
#         dx = dy = dz = (xmax-xmin)/(nx-1)
#         x,y,z = generate_grid(nx, ny, nz, xmin, xmax, ymin, ymax, zmin, zmax)
#         W = trapezoid_weight_matrix(nx, ny, nz)
#         self.assertAlmostEqual(dx, x[1,0,0]-x[0,0,0],10, "dx not as expected.  21 grid points spanning [-10,10] should have dx=1")
#         psi = trueWavefunction(1,x,y,z)
#         psi = normalize_wavefunction(psi, dx, dy, dz, W)
#         V = potential_smoothed(x,y,z,0.1)
#         computed_energy = EnergyUpdate(V, psi, x, y, z, W)
#         true_energy = trueEnergy(1)
#         self.assertAlmostEqual(computed_energy, true_energy,2, "Hamiltonian not giving correct energy from analytic psi") 

    def testPotentialEnergy(self):
        xmin = -8; xmax = 8.25
        ymin = zmin = xmin
        ymax = zmax = xmax
        nx = ny = nz = 81
        dx = dy = dz = (xmax-xmin)/(nx-1)
#         W = trapezoid_weight_matrix(nx, ny, nz)
        W = simpson_weight_matrix(nx, ny, nz)
        x,y,z = generate_grid(nx, ny, nz, xmin, xmax, ymin, ymax, zmin, zmax)
        self.assertAlmostEqual(dx, x[1,0,0]-x[0,0,0],10, "dx not as expected.")
        psi = trueWavefunction(1,x,y,z)
        psi = normalize_wavefunction(psi, dx, dy, dz, W)
        V = potential(x, y, z)
#         V = potential_smoothed(x, y, z, dx/2)
        computed_PE = np.sum(W*psi*V*psi)*dx*dy*dz
        true_PE = -1.0
        potential_error = true_PE - computed_PE
        print('PE error: ', potential_error)
        self.assertAlmostEqual(computed_PE, true_PE, 2, "potential energy not accurate enough")
    
    def testKineticEnergy(self):
        xmin = -8; xmax = 8
        ymin = zmin = xmin
        ymax = zmax = xmax
        nx = ny = nz = 41
        dx = dy = dz = (xmax-xmin)/(nx-1)
        W = trapezoid_weight_matrix(nx, ny, nz)
        x,y,z = generate_grid(nx, ny, nz, xmin, xmax, ymin, ymax, zmin, zmax)
        self.assertAlmostEqual(dx, x[1,0,0]-x[0,0,0],10, "dx not as expected.")
        psi = trueWavefunction(1,x,y,z)
        psi = normalize_wavefunction(psi, dx, dy, dz, W)
        delta_psi = Delta(psi, x, y, z)
        computed_kinetic = -1/2*np.sum(W*psi*delta_psi)*dx*dy*dz
        true_kinetic = 0.5
        kinetic_error = true_kinetic - computed_kinetic
        print('KE error: ', kinetic_error)
        self.assertAlmostEqual(computed_kinetic, true_kinetic, 1, "kinetic not accurate enough")
        
    
    def testRichardsonExtrapolationPE(self):
        xmin = -8; xmax = 8
        ymin = zmin = xmin
        ymax = zmax = xmax
        k=2
        # coarse run
        nxc = nyc = nzc = 41
        W = trapezoid_weight_matrix(nxc, nyc, nzc)
        dxc = dyc = dzc = (xmax-xmin)/(nxc-1)
        x,y,z = generate_grid(nxc, nyc, nzc, xmin, xmax, ymin, ymax, zmin, zmax)
        self.assertAlmostEqual(dxc, x[1,0,0]-x[0,0,0],10, "dx not as expected.")
        psi = trueWavefunction(1,x,y,z)
        psi = normalize_wavefunction(psi, dxc, dyc, dzc, W)
#         V = potential(x, y, z)
        V = potential_smoothed(x, y, z, dxc/2)
        computed_PE_c = np.sum(W*psi*V*psi)*dxc*dyc*dzc
        true_PE = -1.0
        potential_error = true_PE - computed_PE_c
        print('coarse grid PE error: ', potential_error)
        
        # fine run
        nxf = nyf = nzf = k*(nxc-1)+1
        dxf = dyf = dzf = (xmax-xmin)/(nxf-1)
        W = trapezoid_weight_matrix(nxf, nyf, nzf)
        x,y,z = generate_grid(nxf, nyf, nzf, xmin, xmax, ymin, ymax, zmin, zmax)
        self.assertAlmostEqual(dxf, x[1,0,0]-x[0,0,0],10, "dx not as expected.")
        psi = trueWavefunction(1,x,y,z)
        psi = normalize_wavefunction(psi, dxf, dyf, dzf, W)
#         V = potential(x, y, z)
        V = potential_smoothed(x, y, z, dxf/2)
        computed_PE_f = np.sum(W*psi*V*psi)*dxf*dyf*dzf
        true_PE = -1.0
        potential_error = true_PE - computed_PE_f
        print('fine grid PE error: ', potential_error)
        
        # extrpolated value
        extrapolated_PE = (k**2*computed_PE_f - computed_PE_c) / (k**2 - 1)
        extrapolated_error = true_PE - extrapolated_PE
        print('extrapolated PE error: ', extrapolated_error)
#         self.assertEqual(0, 1, "0 not equal to 1")
        
    def testRichardsonExtrapolationKE(self):
        xmin = -8; xmax = 8
        ymin = zmin = xmin
        ymax = zmax = xmax
        k=2
        # coarse run
        nxc = nyc = nzc = 21
        dxc = dyc = dzc = (xmax-xmin)/(nxc-1)
#          Wc = trapezoid_weight_matrix(nxc,nyc,nzc)
        Wc = simpson_weight_matrix(nxc,nyc,nzc)
        x,y,z = generate_grid(nxc, nyc, nzc, xmin, xmax, ymin, ymax, zmin, zmax)
        self.assertAlmostEqual(dxc, x[1,0,0]-x[0,0,0],10, "dx not as expected.")
        psi = trueWavefunction(1,x,y,z)
        psi = normalize_wavefunction(psi, dxc, dyc, dzc, Wc)
        delta_psi = Delta(psi, x, y, z)
        computed_KE_c = -1/2*np.sum(Wc*psi*delta_psi)*dxc*dyc*dzc
        true_KE = 0.5
        kinetic_error = true_KE - computed_KE_c
        print('coarse grid KE error: ', kinetic_error)
        
        # fine run
        nxf = nyf = nzf =  k*(nxc-1)+1  # if nxc odd, use k*(nxc-1)+1.  Else use k*nxc
        dxf = dyf = dzf = (xmax-xmin)/(nxf-1)
#         Wf = trapezoid_weight_matrix(nxf,nyf,nzf)
        Wf = simpson_weight_matrix(nxf,nyf,nzf)
        x,y,z = generate_grid(nxf, nyf, nzf, xmin, xmax, ymin, ymax, zmin, zmax)
        self.assertAlmostEqual(dxf, x[1,0,0]-x[0,0,0],10, "dx not as expected.")
        psi = trueWavefunction(1,x,y,z)
        psi = normalize_wavefunction(psi, dxf, dyf, dzf, Wf)
        delta_psi = Delta(psi, x, y, z)
        computed_KE_f = -1/2*np.sum(Wf*psi*delta_psi)*dxf*dyf*dzf
        true_KE = 0.5
        kinetic_error = true_KE - computed_KE_f
        print('fine grid KE error: ', kinetic_error)
        
        # extrpolated value
        extrapolated_KE = (k**2*computed_KE_f - computed_KE_c) / (k**2 - 1)
        extrapolated_error = true_KE - extrapolated_KE
        print('extrapolated KE error: ', extrapolated_error)
        self.assertEqual(0, 1, "0 not equal to 1")
        
        
        
#     def testGSConvergence(self):
#         xmin = -10; xmax = 10
#         ymin = zmin = xmin
#         ymax = zmax = xmax
#         nx = ny = nz = 20
#         dx = dy = dz = (xmax-xmin)/(nx-1)
#         x,y,z = generate_grid(nx, ny, nz, xmin, xmax, ymin, ymax, zmin, zmax)
#         self.assertAlmostEqual(dx, x[1,0,0]-x[0,0,0],10, "dx not as expected.")
#         psi = trueWavefunction(1,x,y,z)
#         psi = normalize_wavefunction(psi, dx, dy, dz)
#         V = potential_smoothed(x,y,z,0.1)
#         computed_energy_coarse = EnergyUpdate(V, psi, x, y, z)
#         
#         nx = ny = nz = 40
#         dx = dy = dz = (xmax-xmin)/(nx-1)
#         x,y,z = generate_grid(nx, ny, nz, xmin, xmax, ymin, ymax, zmin, zmax)
#         self.assertAlmostEqual(dx, x[1,0,0]-x[0,0,0],10, "dx not as expected.")
#         psi = trueWavefunction(1,x,y,z)
#         psi = normalize_wavefunction(psi, dx, dy, dz)
#         V = potential_smoothed(x,y,z,0.1)
#         computed_energy_fine = EnergyUpdate(V, psi, x, y, z)
#         true_energy = trueEnergy(1)
#         print('coarse error: ', abs(computed_energy_coarse-true_energy))
#         print('fine error:   ', abs(computed_energy_fine-true_energy))
#         self.assertLess(abs(computed_energy_fine-true_energy), abs(computed_energy_coarse-true_energy), "Fine mesh didn't give more accurate energy")
    
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
>>>>>>> refs/remotes/eclipse_auto/master
    unittest.main()