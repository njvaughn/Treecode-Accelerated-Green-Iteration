'''
Created on Feb 4, 2018

@author: nathanvaughn
'''
import unittest
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from cellDataStructure import cell

class TestCell(unittest.TestCase):

    def setUp(self):
        ''' Set up a 3x3x3 cell on [0,1]x[0,1]x[0,1].  Define psi = exp(-r) '''
        x = y = z = np.array([0.0,0.5,1.0])
        psi = np.zeros((3,3,3))
        for i in range(3):
            xt = i/2.0
            for j in range(3):
                yt = j/2.0
                for k in range(3):
                    zt = k/2.0
#                     r = np.sqrt((i*i + j*j + k*k)/4)
                    r = np.sqrt(xt*xt + yt*yt + zt*zt)
                    psi[i,j,k] = np.exp(-r)
        self.Cell = cell(x,y,z,psi)

    def testPsiEvaluation(self):
        self.assertEqual(self.Cell.psi[0,0,0], np.exp(0), "psi not as expected at (0,0,0)")
        self.assertEqual(self.Cell.psi[2,2,2], np.exp(-np.sqrt(3)), "psi not as expected at (1,1,1)")
        self.assertEqual(self.Cell.psi[1,1,1], np.exp(-np.sqrt(3/4)), "psi not as expected at (0.5,0.5,0.5)")
        self.assertEqual(self.Cell.psi[0,1,1], np.exp(-np.sqrt(2/4)), "psi not as expected at (0,0.5,0.5)")
        
    def testDxDyDz(self):
        # test that grid spacing was extracted from x, y, and z correctly
        self.assertEqual(self.Cell.dx, 0.5, "dx not correct")
        self.assertEqual(self.Cell.dy, 0.5, "dy not correct")
        self.assertEqual(self.Cell.dz, 0.5, "dz not correct")

    def testGradientPsi(self):
        # test that the gradient_psi operator returned the expected result, with second order finite difference
        self.Cell.gradient_psi()
        self.assertEqual(np.shape(self.Cell.grad), (3,3,3,3), "gradient not expected shape")
        self.assertEqual(self.Cell.grad[0][1,0,0], (self.Cell.psi[2,0,0]-self.Cell.psi[0,0,0])/(2*self.Cell.dx), "x gradient not as expected along x axis")
        self.assertEqual(self.Cell.grad[2][0,0,1], (self.Cell.psi[0,0,2]-self.Cell.psi[0,0,0])/(2*self.Cell.dz), "z gradient not as expected along z axis")
        # check that secord order forward/backward finite differences used for boundaries
        self.assertEqual(self.Cell.grad[0][0,2,1], (-3/2*self.Cell.psi[0,2,1]+2*self.Cell.psi[1,2,1]-1/2*self.Cell.psi[2,2,1])/(self.Cell.dx), 
                         "gradient not using expected 2nd order forward/backward finite differences along the boundary")


    def testLinearInterpolatorOnLinear(self):
        '''
        set psi to be a linear function, then verify the interpolator performs exactly
        '''
        self.Cell.psi = np.zeros((3,3,3))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    self.Cell.psi[i,j,k] = (i+j+k)/2
        randintegers = np.random.randint(3,size=3)
        i=randintegers[0]; j = randintegers[1]; k = randintegers[2]
        self.Cell.interpolate_for_division()
        self.assertEqual(self.Cell.interpolator((self.Cell.x[i],self.Cell.y[j],self.Cell.z[k])), 
                         self.Cell.psi[i,j,k], 
                         "interpolator didn/'t get one of the original gridpoints correct")
        
        rand3 = np.random.rand(3)
        self.assertAlmostEqual(self.Cell.interpolator(rand3)[0], np.sum(rand3), 15, "interpolator wasn/'t accurate")
        
    def testLinearInterpolatorOnNonlinear(self):
        '''
        now set psi to be nonlinear, verify is performs almost exactly
        '''
        self.Cell.psi = np.zeros((3,3,3))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    self.Cell.psi[i,j,k] = (i*i+j*j+k*k)/4 # r^2 on [0,1]^3 box
        randintegers = np.random.randint(3,size=3)
        i=randintegers[0]; j = randintegers[1]; k = randintegers[2]
        self.Cell.interpolate_for_division()
        self.assertEqual(self.Cell.interpolator((self.Cell.x[i],self.Cell.y[j],self.Cell.z[k])), 
                         self.Cell.psi[i,j,k], 
                         "interpolator didn/'t get one of the original gridpoints correct")
         
#         rand3 = np.random.rand(3)
        rand3 = np.array([0.3,0.55,0.73])
        interpolated_value = self.Cell.interpolator(rand3)[0]
        true_value = np.sum(rand3*rand3)
        self.assertLess((interpolated_value-true_value)/(true_value), 0.3, "interpolator wasn/'t accurate")
        # with this test, checking a random point in the domain and verifying the error isn't very large.  Expect errors
        # up to about 20% using the linear interpolation, depends on the form of psi.  Relative error can blow up if psi small
        
    def testInterpolatorVisually(self):
        def makeheat(coarse_zslice,coarse_data, fine_zslice, fine_data):
            # function that plots slices of the 3D data.  Use to compare original and interpolated, visually check
            plt.figure()
            plt.imshow(coarse_data[:,:,coarse_zslice],interpolation='nearest')
            plt.colorbar()
            plt.title('True Psi on Fine Grid')
            
            plt.figure()
            plt.imshow(fine_data[:,:,fine_zslice],interpolation='nearest')
            plt.colorbar()
            plt.title('Interpolation Error')
            plt.show()
            
        # redefine psi to be the quadratic function
        self.Cell.psi = np.zeros((3,3,3))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    self.Cell.psi[i,j,k] = (i*i+j*j+k*k)/4 # r^2 on [0,1]^3 box
        # create the interpolator
        self.Cell.interpolate_for_division()
        # generate fine mesh (5x5x5 instead of 3x3x3)
        xf = yf = zf = np.array([0,0.25,0.5,0.75,1])
#         xf = yf = zf = np.linspace(0,1,11)
        xfm,yfm,zfm = np.meshgrid(xf,yf,zf,indexing='ij')
        # generate mock fine_psi, assuming the quadratic psi
        fine_psi = xfm*xfm + yfm*yfm + zfm*zfm  
        # generate interpolated psi
        interpolated_psi = self.Cell.interpolator((xfm,yfm,zfm))
        
        if False: # dont want to plot every time I test.
            makeheat(2,fine_psi, 2, interpolated_psi-fine_psi)
        else:
            pass
        
    def testChildren(self):
        children = self.Cell.divide()

        self.assertEqual(children[0,0,0].x[0], self.Cell.x[0], "child 000 isn\'t in the correct corner")
        self.assertEqual(children[0,0,0].y[0], self.Cell.y[0], "child 000 isn\'t in the correct corner")
        self.assertEqual(children[0,0,0].z[0], self.Cell.z[0], "child 000 isn\'t in the correct corner")
        self.assertEqual(children[0,0,0].x[2], self.Cell.x[1], "child 000 isn\'t in the correct corner")
        self.assertEqual(children[0,0,0].y[2], self.Cell.y[1], "child 000 isn\'t in the correct corner")
        self.assertEqual(children[0,0,0].z[2], self.Cell.z[1], "child 000 isn\'t in the correct corner")
        self.assertEqual(np.array_equal(children[0,0,0].psi[::2,::2,::2],self.Cell.psi[0:2,0:2,0:2]), True, 
                         "psi not mapped to 000 correctly")
        
        self.assertEqual(children[1,1,1].x[0], self.Cell.x[1], "child 111 isn\'t in the correct corner")
        self.assertEqual(children[1,1,1].y[0], self.Cell.y[1], "child 111 isn\'t in the correct corner")
        self.assertEqual(children[1,1,1].z[0], self.Cell.z[1], "child 111 isn\'t in the correct corner")
        self.assertEqual(children[1,1,1].x[2], self.Cell.x[2], "child 111 isn\'t in the correct corner")
        self.assertEqual(children[1,1,1].y[2], self.Cell.y[2], "child 111 isn\'t in the correct corner")
        self.assertEqual(children[1,1,1].z[2], self.Cell.z[2], "child 111 isn\'t in the correct corner")
        self.assertEqual(np.array_equal(children[1,1,1].psi[::2,::2,::2],self.Cell.psi[1:3,1:3,1:3]), True, 
                         "psi not mapped to 111 correctly")

        self.assertEqual(children[1,0,1].x[0], self.Cell.x[1], "child 101 isn\'t in the correct corner")
        self.assertEqual(children[1,0,1].y[0], self.Cell.y[0], "child 101 isn\'t in the correct corner")
        self.assertEqual(children[1,0,1].z[0], self.Cell.z[1], "child 101 isn\'t in the correct corner")
        self.assertEqual(children[1,0,1].x[2], self.Cell.x[2], "child 101 isn\'t in the correct corner")
        self.assertEqual(children[1,0,1].y[2], self.Cell.y[1], "child 101 isn\'t in the correct corner")
        self.assertEqual(children[1,0,1].z[2], self.Cell.z[2], "child 101 isn\'t in the correct corner")
        self.assertEqual(np.array_equal(children[1,0,1].psi[::2,::2,::2],self.Cell.psi[1:3,0:2,1:3]), True, 
                         "psi not mapped to 101 correctly")
        
        
    def testDivideCondition(self):
        gradientThreshold = 0.5
        self.Cell.checkDivide(gradientThreshold)
        self.assertEqual(self.Cell.NeedsDividing, True, "divide condition met, but not realized by check_divide")
        
        gradientThreshold = 1.5
        self.Cell.checkDivide(gradientThreshold)
        self.assertEqual(self.Cell.NeedsDividing, False, "divide condition not met, but check_divide returned True")
        
    
    def testLaplacian(self):
        self.Cell.psi = np.zeros((3,3,3))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    self.Cell.psi[i,j,k] = (i*i+j*j+k*k)/4 # r^2 on [0,1]^3 box
        
        self.Cell.computeLaplacian()
#         self.assertEqual(np.array_equal(np.ones((3,3,3))*6,self.Cell.Laplacian), True, 
#                          "Computed Laplacian not correct for quadratic function")
        self.assertEqual(6,self.Cell.Laplacian, 
                         "Computed Laplacian not correct for quadratic function")
        
    def testPotential(self):
        self.Cell.evaluatePotential()
        r = np.sqrt(self.Cell.x[1]**2 + self.Cell.y[1]**2 + self.Cell.z[1]**2)
        self.assertEqual(self.Cell.Potential, -1/r)
        
        
        
      
        
        
        
        
          

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()