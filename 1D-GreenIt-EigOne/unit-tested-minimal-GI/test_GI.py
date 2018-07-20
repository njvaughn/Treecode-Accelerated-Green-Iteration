<<<<<<< HEAD
'''
Created on Jan 17, 2018

@author: nathanvaughn
'''
import unittest
from GI_minimal import GIrun
import numpy as np


class testGI(unittest.TestCase):
    '''
    For testing a Green Iterations Run
    '''
    @classmethod
    def setUpClass(self):
        self.testRun = GIrun(model_name="Poschl-Teller",nx=100,xmin=-10,xmax=10,D=4)
        
    def testGIRunClass(self):
        self.assertEqual(self.testRun.nx, 100, "Incorrect Number of Gridpoints")
        self.assertEqual(self.testRun.xmin, -10, "Incorrect left endpoint")
        self.assertEqual(self.testRun.xmax, 10, "Incorrect right endpoint")
        self.assertEqual(self.testRun.D, 4, "Incorrect Well Scaling")
        self.assertEqual(self.testRun.true_energy(0,4), -8.0)
        self.assertEqual(self.testRun.potential(0, 4), -10, "incorrect well depth")
        self.assertEqual(self.testRun.dx, 0.2, "dx not correct")
        self.assertEqual(self.testRun.xgrid[0], self.testRun.xmin + self.testRun.dx/2, "xgrid not set up properly")
                       
    def testHamiltonian(self):
        '''
        Test the setup of the hamiltonian.   Right shape, right symmetry, etc.
        '''
        hamiltonian = self.testRun.Hamiltonian()
        self.assertEqual(np.shape(hamiltonian), (self.testRun.nx,self.testRun.nx), "hamiltonian shape not right")
        self.testRun.V = np.zeros(self.testRun.nx)
        hamiltonian = self.testRun.Hamiltonian()
        self.assertEqual(hamiltonian[0,0], -2*hamiltonian[0,1], "verifying laplace discretization")
        self.assertEqual(hamiltonian[-1,-1],hamiltonian[0,0])

        '''
        test that the Hamiltonian returns exact result for linear and quadratic 
        functions with constant potential.  WLOG, check the 50th point.
        '''
        self.assertEqual(np.dot(hamiltonian,5.5*self.testRun.xgrid)[50], 0.0, "-laplacian should return 0 for linear function")
        self.assertAlmostEqual(np.dot(hamiltonian,5.5*self.testRun.xgrid*self.testRun.xgrid)[50], -5.5, 10, "-laplacian should return 5.5 for this quadrati")
       
    def testGreensFunction(self):
        self.assertNotEqual(self.testRun.G(x = 1.0,z = -9), self.testRun.G(x = -1.0,z = -9), "Green's function is not even")
        self.assertEqual(self.testRun.G(0,-1.0),-1/(2*np.sqrt(2)) )
        
    def testConvolutionOperator(self):
        z = -9.0
        convolution_operator = self.testRun.ConvolutionOperator(z)
        self.assertEqual(np.shape(convolution_operator),(self.testRun.nx,self.testRun.nx),"Convolution operator not right shape")
        self.testRun.V = np.ones(self.testRun.nx)
        self.assertNotEqual(convolution_operator[0,10],convolution_operator[10,0],"Convolution operator shouldn't be symmetric (for generic potentials)")
        convolution_operator = self.testRun.ConvolutionOperator(z)
        self.assertEqual(convolution_operator[0,10],convolution_operator[10,0],"Green's function is symmetric")
        
    def testNormalize(self):
        random = np.random.rand(self.testRun.nx)
        normalized = self.testRun.normalize(random)
        self.assertAlmostEqual(np.dot(normalized,normalized)*self.testRun.dx, 1.0, 13, "normalization not working")

    def testComputeGroundStateEnergy(self):
        testRunFine = GIrun(model_name="Poschl-Teller",nx=300,xmin=-10,xmax=10,D=4)
        ground_state_energy = testRunFine.compute_energy(eig_tol=1e-8,z_in=-7,psi_in=np.random.rand(testRunFine.nx))
        self.assertAlmostEqual(ground_state_energy,self.testRun.true_energy(0, testRunFine.D),2,"Calculated energy not accurate.")

    def testMeshConvergence(self):
        '''
        Test a sequence of three energy calculations.  The grids are refined by a factor of 2
        each time, and the test checks for second order convergence.  This should pass given
        the 2nd order accuracy of the midpoint method and the second order accuracy of the
        discretized Hamiltonian.
        '''
        testRunCoarse = GIrun(model_name="Poschl-Teller",nx=300,xmin=-10,xmax=10,D=4)
        coarse_energy = testRunCoarse.compute_energy(eig_tol=1e-14,z_in=-7,psi_in=np.random.rand(testRunCoarse.nx))
        testRunMid = GIrun(model_name="Poschl-Teller",nx=600,xmin=-10,xmax=10,D=4)
        mid_energy = testRunMid.compute_energy(eig_tol=1e-14,z_in=-7,psi_in=np.random.rand(testRunMid.nx))
        testRunFine = GIrun(model_name="Poschl-Teller",nx=1200,xmin=-10,xmax=10,D=4)
        fine_energy = testRunFine.compute_energy(eig_tol=1e-14,z_in=-7,psi_in=np.random.rand(testRunFine.nx))
        
        coarse_error = coarse_energy - self.testRun.true_energy(0, self.testRun.D)
        mid_error   = mid_energy   - self.testRun.true_energy(0, self.testRun.D)
        fine_error   = fine_energy   - self.testRun.true_energy(0, self.testRun.D)
        
        self.assertLess(abs(fine_error), abs(mid_error), "error didn't decrease as mesh spaing decreased from mid to fine.")
        self.assertLess(abs(mid_error), abs(coarse_error), "error didn't decrease as mesh spaing decreased from coarse to mid.")
        self.assertAlmostEqual(16*fine_error/coarse_error, 1, 2, "fine error wasn't roughly 1/16 of coarse error, indicating not 2nd order convergence.")
        self.assertAlmostEqual(coarse_error/mid_error, mid_error/fine_error, 1, "Convergence rates between coarse-mid and mid-fine do not match.")
        
=======
'''
Created on Jan 17, 2018

@author: nathanvaughn
'''
import unittest
from GI_minimal import GIrun
import numpy as np


class testGI(unittest.TestCase):
    '''
    For testing a Green Iterations Run
    '''
    @classmethod
    def setUpClass(self):
        self.testRun = GIrun(model_name="Poschl-Teller",nx=100,xmin=-10,xmax=10,D=4)
        
    def testGIRunClass(self):
        self.assertEqual(self.testRun.nx, 100, "Incorrect Number of Gridpoints")
        self.assertEqual(self.testRun.xmin, -10, "Incorrect left endpoint")
        self.assertEqual(self.testRun.xmax, 10, "Incorrect right endpoint")
        self.assertEqual(self.testRun.D, 4, "Incorrect Well Scaling")
        self.assertEqual(self.testRun.true_energy(0,4), -8.0)
        self.assertEqual(self.testRun.potential(0, 4), -10, "incorrect well depth")
        self.assertEqual(self.testRun.dx, 0.2, "dx not correct")
        self.assertEqual(self.testRun.xgrid[0], self.testRun.xmin + self.testRun.dx/2, "xgrid not set up properly")
                       
    def testHamiltonian(self):
        '''
        Test the setup of the hamiltonian.   Right shape, right symmetry, etc.
        '''
        hamiltonian = self.testRun.Hamiltonian()
        self.assertEqual(np.shape(hamiltonian), (self.testRun.nx,self.testRun.nx), "hamiltonian shape not right")
        self.testRun.V = np.zeros(self.testRun.nx)
        hamiltonian = self.testRun.Hamiltonian()
        self.assertEqual(hamiltonian[0,0], -2*hamiltonian[0,1], "verifying laplace discretization")
        self.assertEqual(hamiltonian[-1,-1],hamiltonian[0,0])

        '''
        test that the Hamiltonian returns exact result for linear and quadratic 
        functions with constant potential.  WLOG, check the 50th point.
        '''
        self.assertEqual(np.dot(hamiltonian,5.5*self.testRun.xgrid)[50], 0.0, "-laplacian should return 0 for linear function")
        self.assertAlmostEqual(np.dot(hamiltonian,5.5*self.testRun.xgrid*self.testRun.xgrid)[50], -5.5, 10, "-laplacian should return 5.5 for this quadrati")
       
    def testGreensFunction(self):
        self.assertNotEqual(self.testRun.G(x = 1.0,z = -9), self.testRun.G(x = -1.0,z = -9), "Green's function is not even")
        self.assertEqual(self.testRun.G(0,-1.0),-1/(2*np.sqrt(2)) )
        
    def testConvolutionOperator(self):
        z = -9.0
        convolution_operator = self.testRun.ConvolutionOperator(z)
        self.assertEqual(np.shape(convolution_operator),(self.testRun.nx,self.testRun.nx),"Convolution operator not right shape")
        self.testRun.V = np.ones(self.testRun.nx)
        self.assertNotEqual(convolution_operator[0,10],convolution_operator[10,0],"Convolution operator shouldn't be symmetric (for generic potentials)")
        convolution_operator = self.testRun.ConvolutionOperator(z)
        self.assertEqual(convolution_operator[0,10],convolution_operator[10,0],"Green's function is symmetric")
        
    def testNormalize(self):
        random = np.random.rand(self.testRun.nx)
        normalized = self.testRun.normalize(random)
        self.assertAlmostEqual(np.dot(normalized,normalized)*self.testRun.dx, 1.0, 13, "normalization not working")

    def testComputeGroundStateEnergy(self):
        testRunFine = GIrun(model_name="Poschl-Teller",nx=300,xmin=-10,xmax=10,D=4)
        ground_state_energy = testRunFine.compute_energy(eig_tol=1e-8,z_in=-7,psi_in=np.random.rand(testRunFine.nx))
        self.assertAlmostEqual(ground_state_energy,self.testRun.true_energy(0, testRunFine.D),2,"Calculated energy not accurate.")

    def testMeshConvergence(self):
        '''
        Test a sequence of three energy calculations.  The grids are refined by a factor of 2
        each time, and the test checks for second order convergence.  This should pass given
        the 2nd order accuracy of the midpoint method and the second order accuracy of the
        discretized Hamiltonian.
        '''
        testRunCoarse = GIrun(model_name="Poschl-Teller",nx=300,xmin=-10,xmax=10,D=4)
        coarse_energy = testRunCoarse.compute_energy(eig_tol=1e-14,z_in=-7,psi_in=np.random.rand(testRunCoarse.nx))
        testRunMid = GIrun(model_name="Poschl-Teller",nx=600,xmin=-10,xmax=10,D=4)
        mid_energy = testRunMid.compute_energy(eig_tol=1e-14,z_in=-7,psi_in=np.random.rand(testRunMid.nx))
        testRunFine = GIrun(model_name="Poschl-Teller",nx=1200,xmin=-10,xmax=10,D=4)
        fine_energy = testRunFine.compute_energy(eig_tol=1e-14,z_in=-7,psi_in=np.random.rand(testRunFine.nx))
        
        coarse_error = coarse_energy - self.testRun.true_energy(0, self.testRun.D)
        mid_error   = mid_energy   - self.testRun.true_energy(0, self.testRun.D)
        fine_error   = fine_energy   - self.testRun.true_energy(0, self.testRun.D)
        
        self.assertLess(abs(fine_error), abs(mid_error), "error didn't decrease as mesh spaing decreased from mid to fine.")
        self.assertLess(abs(mid_error), abs(coarse_error), "error didn't decrease as mesh spaing decreased from coarse to mid.")
        self.assertAlmostEqual(16*fine_error/coarse_error, 1, 2, "fine error wasn't roughly 1/16 of coarse error, indicating not 2nd order convergence.")
        self.assertAlmostEqual(coarse_error/mid_error, mid_error/fine_error, 1, "Convergence rates between coarse-mid and mid-fine do not match.")
        
>>>>>>> refs/remotes/eclipse_auto/master
    