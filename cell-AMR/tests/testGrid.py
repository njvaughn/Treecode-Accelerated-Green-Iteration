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
from hydrogenPotential import trueWavefunction





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
    

        
    def testRefinedGridVisualization(self):
        original_x,original_y,original_z,original_psi = self.grid.extractGridFromCellArray()
        psiGradientThreshold = 0.2
        self.grid.GridRefinement(psiGradientThreshold)
        self.grid.GridRefinement(psiGradientThreshold)
#         self.grid.GridRefinement(psiGradientThreshold)
        refined_x,refined_y,refined_z,refined_psi = self.grid.extractGridFromCellArray()
        
        # extract slices
        xOriginalSlice = []
        yOriginalSlice = []
        psiOriginalSlice = []
        for index in range(len(original_x)):
            if ((original_z[index] > 0.1) and (original_z[index] < 1.5) ):  # there is a slice of midpoints that goes through z=1
                xOriginalSlice.append(original_x[index])
                yOriginalSlice.append(original_y[index])
                psiOriginalSlice.append(original_psi[index])
        xRefinedSlice = []
        yRefinedSlice = []
        psiRefinedSlice = []
        for index in range(len(refined_x)):
            if ((refined_z[index] > 0.1) and (refined_z[index] < 1.5) ):  # there is a slice of midpoints that goes through z=1
                xRefinedSlice.append(refined_x[index])
                yRefinedSlice.append(refined_y[index])
                psiRefinedSlice.append(refined_psi[index])
        
        def makescatter(xCoarse,yCoarse, psiCoarse, xFine, yFine, psiFine):
            # plot a fine grained psi
            x,y = np.meshgrid( np.linspace(-8,8,100), np.linspace(-8,8,100), indexing='ij' )
            z = np.ones(np.shape(x))
            psi = trueWavefunction(1, x, y, z)
            plt.figure()
            plt.imshow(psi,interpolation='nearest',extent=[-8,8,-8,8],origin="lower")
            plt.clim(0,0.75)
            plt.colorbar()
            
            # function that plots slices of the 3D data.  Use to compare original and interpolated, visually check
#             plt.figure()
#             plt.scatter(xCoarse,yCoarse,c=psiCoarse,s=1)
            plt.scatter(xCoarse,yCoarse,c='white',s=1)
#             plt.scatter(xCoarse,yCoarse,c=psiCoarse,interpolation='nearest')
#             plt.clim(0,0.75)
#             plt.colorbar()
            plt.title('Psi on Original Grid')
            
            plt.figure()
            plt.imshow(psi,interpolation='nearest',extent=[-8,8,-8,8],origin="lower")
            plt.clim(0,0.75)
            plt.colorbar()
            plt.scatter(xFine,yFine,c='white',s=1)
            plt.title('Psi on Refined Grid')
            plt.show()
            
        
        makescatter(xOriginalSlice,yOriginalSlice,psiOriginalSlice,xRefinedSlice,yRefinedSlice,psiRefinedSlice)
        



if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()