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
        
    def testPotentialCalculation(self):
        self.xmin = self.ymin = self.zmin = -40
        self.xmax = self.ymax = self.zmax =  40
        self.nx = self.ny = self.nz = 12
        self.grid = Grid(self.xmin,self.xmax,self.ymin,self.ymax,self.zmin,self.zmax,self.nx,self.ny,self.nz)
#         W3 = self.grid.cells[0].simpson_weight_matrix(3,3,3)
        psiVariationThreshold = 0.05
        levels = 9
        for level in range(levels):
            if level > 0:
                self.grid.GridRefinement(psiVariationThreshold)
#                 psiGradientThreshold += 0.1
            computedPotential = 0.0
            sumPsiSquaredDxDyDz = 0.0
            for index in range(len(self.grid.cells)):
                sumPsiSquaredDxDyDz += self.grid.cells[index].psi[1,1,1]**2*self.grid.cells[index].volume
#                 sumPsiSquaredDxDyDz += np.sum(W3*self.grid.cells[index].psi**2)*self.grid.cells[index].volume
            
            for index in range(len(self.grid.cells)):
                xm,ym,zm = np.meshgrid(self.grid.cells[index].x,self.grid.cells[index].y,
                            self.grid.cells[index].z,indexing='ij')
                self.grid.cells[index].psi = trueWavefunction(1, xm, ym, zm)
                self.grid.cells[index].evaluatePotential_MidpointMethod()
#                 self.grid.cells[index].evaluatePotential_SimpsonMethod()
                computedPotential += self.grid.cells[index].Potential
            computedPotential = computedPotential/sumPsiSquaredDxDyDz
            print('Pass ',level,': ', len(self.grid.cells),' cells. Computed Potential: ', computedPotential)
            # check psi at some random point that is one of the children
            if level > 0:
                idx = np.random.randint(self.nx*self.nz*self.nz,len(self.grid.cells))
                x = self.grid.cells[idx].x[1]; y = self.grid.cells[idx].y[1]; z = self.grid.cells[idx].z[1]
                self.assertEqual(self.grid.cells[idx].psi[1,1,1], trueWavefunction(1, x, y, z), 
                                 "wavefunction not expected value at midpoint after divisions")
                x = self.grid.cells[idx].x[2]; y = self.grid.cells[idx].y[0]; z = self.grid.cells[idx].z[1]
                self.assertEqual(self.grid.cells[idx].psi[2,0,1], trueWavefunction(1, x, y, z), 
                                 "wavefunction not expected value on boundary after divisions")
                
    def pass_testKineticCalculation(self):
        self.xmin = self.ymin = self.zmin = -16
        self.xmax = self.ymax = self.zmax =  16
        self.nx = self.ny = self.nz = 12
        self.grid = Grid(self.xmin,self.xmax,self.ymin,self.ymax,self.zmin,self.zmax,self.nx,self.ny,self.nz)
#         W3 = self.grid.cells[0].simpson_weight_matrix(3,3,3)
        psiVariationThreshold = 0.25
        levels = 5
        for level in range(levels):
            if level > 0:
                self.grid.GridRefinement(psiVariationThreshold)
#                 psiGradientThreshold += 0.1
            computedKinetic = 0.0
            sumPsiSquaredDxDyDz = 0.0
            for index in range(len(self.grid.cells)):
                sumPsiSquaredDxDyDz += self.grid.cells[index].psi[1,1,1]**2*self.grid.cells[index].volume
#                 sumPsiSquaredDxDyDz += np.sum(W3*self.grid.cells[index].psi**2)*self.grid.cells[index].volume
            
            for index in range(len(self.grid.cells)):
                xm,ym,zm = np.meshgrid(self.grid.cells[index].x,self.grid.cells[index].y,
                            self.grid.cells[index].z,indexing='ij')
                self.grid.cells[index].psi = trueWavefunction(1, xm, ym, zm)
                self.grid.cells[index].evaluateKinetic_MidpointMethod()

                computedKinetic += self.grid.cells[index].Kinetic
            computedKinetic = computedKinetic/sumPsiSquaredDxDyDz
            print('Pass ',level,': ', len(self.grid.cells),' cells. Computed Kinetic: ', computedKinetic)
            
   


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()