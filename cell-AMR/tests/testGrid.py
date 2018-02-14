'''
Created on Feb 7, 2018

@author: nathanvaughn
'''
import unittest
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from gridUtilities import Grid
from hydrogenPotential import trueWavefunction, potential





class TestGrid(unittest.TestCase):


    def setUp(self):
        """
        Goal:  set up a grid that will allow me to initialize 8x8x8 cells.  
        """
        self.xmin = self.ymin = self.zmin = -8
        self.xmax = self.ymax = self.zmax =  8
        self.nx = self.ny = self.nz = 8
        self.grid = Grid(self.xmin,self.xmax,self.ymin,self.ymax,self.zmin,self.zmax,self.nx,self.ny,self.nz)


    def tearDown(self):
        self.grid = None

    def testGridShape(self):
        self.assertEqual(np.shape(self.grid.cells), (self.nx*self.ny*self.nz,), "Grid wasn't the expected shape")
    
    def testCellCoordinates(self):
        def index(i,j,k):
            return self.ny*self.nz*i + self.nz*j + k
        # indexing    ny*nz*i + nz*j + k
        self.assertEqual(self.grid.cells[index(0,0,0)].x[0], self.xmin, "corner x didn't have expexted value")
        self.assertEqual(self.grid.cells[index(0,0,0)].x[2], self.grid.cells[index(1,0,0)].x[0], "adjacent cells not sharing x value")
        self.assertEqual((self.grid.cells[index(3,3,3)].x[2],self.grid.cells[index(3,3,3)].y[2],self.grid.cells[index(3,3,3)].z[2]),
                          (0,0,0), "expect the 333 cell's corner to be at origin")
        self.assertEqual((self.grid.cells[index(4,5,6)].x[1],self.grid.cells[index(4,5,6)].y[1],self.grid.cells[index(4,5,6)].z[1]),
                          (1,3,5), "expect the 456 cell's midpoint to be at (1,3,5)")
        self.assertEqual(self.grid.cells[index(0,0,0)].dx, 1, "Cell dx not equal to 1, as expected from dividing [-8,8] into 8 cells of width 2dx")
#     
    def testCellPsiValues(self):
        def index(i,j,k):
            return self.ny*self.nz*i + self.nz*j + k
        self.assertEqual(self.grid.cells[index(3,3,3)].psi[2,2,2], trueWavefunction(1, 0, 0, 0), "psi at origin not correct")
        self.assertEqual(self.grid.cells[index(1,4,1)].psi[1,1,1], trueWavefunction(1, self.grid.cells[index(1,4,1)].x[1],
                         self.grid.cells[index(1,4,1)].y[1], self.grid.cells[index(1,4,1)].z[1]), "psi at cell 141 midpoint not correct")
    

        
#     def testRefinedGridVisualization(self):
#         original_x,original_y,original_z,original_psi = self.grid.extractGridFromCellArray()
#         psiGradientThreshold = 0.2
#         self.grid.GridRefinement(psiGradientThreshold)
#         self.grid.GridRefinement(psiGradientThreshold)
# #         self.grid.GridRefinement(psiGradientThreshold)
#         refined_x,refined_y,refined_z,refined_psi = self.grid.extractGridFromCellArray()
#         
#         # extract slices
#         xOriginalSlice = []
#         yOriginalSlice = []
#         psiOriginalSlice = []
#         for index in range(len(original_x)):
#             if ((original_z[index] > 0.1) and (original_z[index] < 1.5) ):  # there is a slice of midpoints that goes through z=1
#                 xOriginalSlice.append(original_x[index])
#                 yOriginalSlice.append(original_y[index])
#                 psiOriginalSlice.append(original_psi[index])
#         xRefinedSlice = []
#         yRefinedSlice = []
#         psiRefinedSlice = []
#         for index in range(len(refined_x)):
#             if ((refined_z[index] > 0.1) and (refined_z[index] < 1.5) ):  # there is a slice of midpoints that goes through z=1
#                 xRefinedSlice.append(refined_x[index])
#                 yRefinedSlice.append(refined_y[index])
#                 psiRefinedSlice.append(refined_psi[index])
#         
#         def makescatter(xCoarse,yCoarse, psiCoarse, xFine, yFine, psiFine):
#             # plot a fine grained psi
#             x,y = np.meshgrid( np.linspace(-8,8,100), np.linspace(-8,8,100), indexing='ij' )
#             z = np.ones(np.shape(x))
#             psi = trueWavefunction(1, x, y, z)
#             plt.figure()
#             plt.imshow(psi,interpolation='nearest',extent=[-8,8,-8,8],origin="lower")
#             plt.clim(0,0.75)
#             plt.colorbar()
#             
# 
#             plt.scatter(xCoarse,yCoarse,c='white',s=1)
#             plt.title('Psi on Original Grid')
#             
#             plt.figure()
#             plt.imshow(psi,interpolation='nearest',extent=[-8,8,-8,8],origin="lower")
#             plt.clim(0,0.75)
#             plt.colorbar()
#             plt.scatter(xFine,yFine,c='white',s=1)
#             plt.title('Psi on Refined Grid')
#             plt.show()
#             
#         """ uncomment to show plots of the refined grid """
# #         makescatter(xOriginalSlice,yOriginalSlice,psiOriginalSlice,xRefinedSlice,yRefinedSlice,psiRefinedSlice)
        
                
    def testNormalization(self):
        W = self.grid.MidpointWeightMatrix()
#         preNormalizationSum = 0.0
#         for index in range(len(self.grid.cells)):
#             preNormalizationSum += self.grid.cells[index].psi[1,1,1]**2*self.grid.cells[index].volume
        postNormalizationSum = 0.0
        self.grid.normalizePsi(W)
        for index in range(len(self.grid.cells)):
            postNormalizationSum += np.sum(W*self.grid.cells[index].psi**2*self.grid.cells[index].volume)
        self.assertAlmostEqual(1.0, postNormalizationSum,14, "wavefunction normalization error for midpoint.")
        
        postNormalizationSum = 0.0
        W = self.grid.SimpsonWeightMatrix()
        self.grid.normalizePsi(W)
        for index in range(len(self.grid.cells)):
            postNormalizationSum += np.sum(W*self.grid.cells[index].psi**2*self.grid.cells[index].volume)
        self.assertAlmostEqual(1.0, postNormalizationSum,14, "wavefunction normalization error for simpson.")
       
    def testMidpointWeightMatrix(self): 
        W = self.grid.MidpointWeightMatrix()
        self.assertEqual(np.sum(W), 1.0, "midpoint weights don't sum to 1")    
        self.assertEqual(np.max(W),1.0,"max value not 1.")    
        self.assertEqual(np.min(W),0.0,"max value not 1.")  
        
    def testSimpsonWeightMatrix(self): 
        W = self.grid.SimpsonWeightMatrix()
        self.assertEqual(np.max(W), 64/27/8, "max not as expected")
        self.assertEqual(np.argmax(W), 13, "max value of W not in center")   
        self.assertEqual(W[0,0,0], W[2,2,2], "corners not equal") 
        self.assertEqual(np.sum(W), 1, "simpson weights dont sum to 1")
        
    def testSimpsonIntegration(self):
        def setFakeWavefunction(self):
            '''
            Generate fake wavefunctions used to test the integrators
            '''
            for index in range(len(self.grid.cells)): 
                    xm,ym,zm = np.meshgrid(self.grid.cells[index].x,self.grid.cells[index].y,
                                self.grid.cells[index].z,indexing='ij')
                    self.grid.cells[index].psi = xm**2 + ym**2 + zm**2  # quadratic
                
        def evaluateIntegral(self,W):
            I = 0.0
            for index in range(len(self.grid.cells)):
                I += np.sum( W*self.grid.cells[index].psi*self.grid.cells[index].volume )
            return I
            
        self.xmin = self.ymin = self.zmin = -2
        self.xmax = self.ymax = self.zmax =  2
        self.nx = self.ny = self.nz = 8
        self.grid = Grid(self.xmin,self.xmax,self.ymin,self.ymax,self.zmin,self.zmax,self.nx,self.ny,self.nz)
        
        setFakeWavefunction(self)
#         W = self.grid.MidpointWeightMatrix()
        W = self.grid.SimpsonWeightMatrix()
        Integral = evaluateIntegral(self,W)
#         print('Integral = ', Integral)
        self.assertEqual(Integral, 256, "Simpson not integrating quadratic exactly")
        
        
        
    def passtestKineticCalculation(self):
        self.xmin = self.ymin = self.zmin = -8
        self.xmax = self.ymax = self.zmax =  8
        self.nx = self.ny = self.nz = 14
        self.grid = Grid(self.xmin,self.xmax,self.ymin,self.ymax,self.zmin,self.zmax,self.nx,self.ny,self.nz)
#         W = self.grid.MidpointWeightMatrix()
        W = self.grid.SimpsonWeightMatrix()

        psiVariationThreshold = 0.05
        levels = 6
        for level in range(levels):
            if level > 0:
                self.grid.GridRefinement(psiVariationThreshold)
                self.grid.setExactWavefunction()
            self.grid.normalizePsi(W)
            self.grid.computeKinetic(W)
            print('Pass ',level,': ', len(self.grid.cells),' cells. Kinetic error: ', 0.5-self.grid.Kinetic)
            
            
    def passtestPotentialCalculation(self):
        self.xmin = self.ymin = self.zmin = -12
        self.xmax = self.ymax = self.zmax =  12
        self.nx = self.ny = self.nz = 14
        self.grid = Grid(self.xmin,self.xmax,self.ymin,self.ymax,self.zmin,self.zmax,self.nx,self.ny,self.nz)
        W = self.grid.MidpointWeightMatrix()
#         W = self.grid.SimpsonWeightMatrix()
        psiVariationThreshold = 0.05
        epsilon = 0.0000000001
        levels = 6
        for level in range(levels):
            if level > 0:
                self.grid.GridRefinement(psiVariationThreshold)
                self.grid.setExactWavefunction()
            self.grid.normalizePsi(W)
            self.grid.computePotential(W,epsilon)
            
            print('Pass ',level,': ', len(self.grid.cells),' cells. Potential error: ', -1.0-self.grid.Potential)


    def passtestKineticAndPotential(self):  
        print()
        self.xmin = self.ymin = self.zmin = -8
        self.xmax = self.ymax = self.zmax =  8
        self.nx = self.ny = self.nz = 16
        self.grid = Grid(self.xmin,self.xmax,self.ymin,self.ymax,self.zmin,self.zmax,self.nx,self.ny,self.nz)
#         W = self.grid.MidpointWeightMatrix()
        W = self.grid.SimpsonWeightMatrix()
        psiVariationThreshold = 0.025
        epsilon = 0.025
        levels = 6
        for level in range(levels):
            if level > 0:
                self.grid.GridRefinement(psiVariationThreshold)
                self.grid.setExactWavefunction()
            self.grid.normalizePsi(W)
            self.grid.computePotential(W,epsilon)
            self.grid.computeKinetic(W)
            print('Pass ',level,': ', len(self.grid.cells),' cells.')
            print('Kinetic error: ', 0.5-self.grid.Kinetic)
            print('Potential error: ', -1.0-self.grid.Potential,'\n')
   


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()