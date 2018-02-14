'''
Created on Feb 7, 2018

@author: nathanvaughn
'''
import unittest
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from meshUtilities import Mesh
from hydrogenPotential import trueWavefunction, potential





class TestMesh(unittest.TestCase):


    def setUp(self):
        """
        Set up a mesh that will be used in many of the following tests.  
        """
        self.xmin = self.ymin = self.zmin = -8
        self.xmax = self.ymax = self.zmax =  8
        self.nx = self.ny = self.nz = 8
        self.mesh = Mesh(self.xmin,self.xmax,self.ymin,self.ymax,self.zmin,self.zmax,self.nx,self.ny,self.nz,"analytic")


    def tearDown(self):
        self.mesh = None

    def testMeshShape(self):
        '''
        Verify the shape of the mesh-of-cells
        '''
        self.assertEqual(np.shape(self.mesh.cells), (self.nx*self.ny*self.nz,), "Mesh wasn't the expected shape")
    
    def testCellCoordinates(self):
        '''
        Verify that the mesh is getting organized into cells correctly
        '''
        def index(i,j,k):
            return self.ny*self.nz*i + self.nz*j + k
        # indexing scheme:    ny*nz*i + nz*j + k
        self.assertEqual(self.mesh.cells[index(0,0,0)].x[0], self.xmin, "corner x didn't have expexted value")
        self.assertEqual(self.mesh.cells[index(0,0,0)].x[2], self.mesh.cells[index(1,0,0)].x[0], "adjacent cells not sharing x value")
        self.assertEqual((self.mesh.cells[index(3,3,3)].x[2],self.mesh.cells[index(3,3,3)].y[2],self.mesh.cells[index(3,3,3)].z[2]),
                          (0,0,0), "expect the 333 cell's corner to be at origin")
        self.assertEqual((self.mesh.cells[index(4,5,6)].x[1],self.mesh.cells[index(4,5,6)].y[1],self.mesh.cells[index(4,5,6)].z[1]),
                          (1,3,5), "expect the 456 cell's midpoint to be at (1,3,5)")
        self.assertEqual(self.mesh.cells[index(0,0,0)].dx, 1, "Cell dx not equal to 1, as expected from dividing [-8,8] into 8 cells of width 2dx")
#     
    def testCellPsiValues(self):
        '''
        Verify that the wavefunction is getting mapped to the mesh as expected when the initial wavefunction is set to analytic. 
        Verify that the random wavefunction option works. 
        '''
        def index(i,j,k):
            return self.ny*self.nz*i + self.nz*j + k
        self.assertEqual(self.mesh.cells[index(3,3,3)].psi[2,2,2], trueWavefunction(1, 0, 0, 0), "psi at origin not correct")
        self.assertEqual(self.mesh.cells[index(1,4,1)].psi[1,1,1], trueWavefunction(1, self.mesh.cells[index(1,4,1)].x[1],
                         self.mesh.cells[index(1,4,1)].y[1], self.mesh.cells[index(1,4,1)].z[1]), "psi at cell 141 midpoint not correct")
        
        self.mesh = Mesh(self.xmin,self.xmax,self.ymin,self.ymax,self.zmin,self.zmax,self.nx,self.ny,self.nz,"random")
        self.assertNotEqual(self.mesh.cells[index(3,3,3)].psi[2,2,2], trueWavefunction(1, 0, 0, 0), "psi at origin not correct")
        self.assertNotEqual(self.mesh.cells[index(1,4,1)].psi[1,1,1], trueWavefunction(1, self.mesh.cells[index(1,4,1)].x[1],
                         self.mesh.cells[index(1,4,1)].y[1], self.mesh.cells[index(1,4,1)].z[1]), "psi at cell 141 midpoint not correct")
         
    def passtestRefinedMeshVisualization(self):
        '''
        Manually visualize the mesh refinement.  Pass this test unless manually inspecting the figures.  
        '''
        original_x,original_y,original_z,original_psi = self.mesh.extractMeshFromCellArray()
        psiGradientThreshold = 0.05
        self.mesh.MeshRefinement(psiGradientThreshold)
        self.mesh.MeshRefinement(psiGradientThreshold)
#         self.mesh.MeshRefinement(psiGradientThreshold)
        refined_x,refined_y,refined_z,refined_psi = self.mesh.extractMeshFromCellArray()
         
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
             
 
            plt.scatter(xCoarse,yCoarse,c='white',s=1)
            plt.title('Psi on Original Mesh')
             
            plt.figure()
            plt.imshow(psi,interpolation='nearest',extent=[-8,8,-8,8],origin="lower")
            plt.clim(0,0.75)
            plt.colorbar()
            plt.scatter(xFine,yFine,c='white',s=1)
            plt.title('Psi on Refined Mesh')
            plt.show()
             
        makescatter(xOriginalSlice,yOriginalSlice,psiOriginalSlice,xRefinedSlice,yRefinedSlice,psiRefinedSlice)
                    
    def testNormalization(self):
        '''
        Test that the normalization function does in fact work, regardless of the weight matrix chosen.
        '''
        W = self.mesh.MidpointWeightMatrix()
#         preNormalizationSum = 0.0
#         for index in range(len(self.mesh.cells)):
#             preNormalizationSum += self.mesh.cells[index].psi[1,1,1]**2*self.mesh.cells[index].volume
        postNormalizationSum = 0.0
        self.mesh.normalizePsi(W)
        for index in range(len(self.mesh.cells)):
            postNormalizationSum += np.sum(W*self.mesh.cells[index].psi**2*self.mesh.cells[index].volume)
        self.assertAlmostEqual(1.0, postNormalizationSum,14, "wavefunction normalization error for midpoint.")
        
        postNormalizationSum = 0.0
        W = self.mesh.SimpsonWeightMatrix()
        self.mesh.normalizePsi(W)
        for index in range(len(self.mesh.cells)):
            postNormalizationSum += np.sum(W*self.mesh.cells[index].psi**2*self.mesh.cells[index].volume)
        self.assertAlmostEqual(1.0, postNormalizationSum,14, "wavefunction normalization error for simpson.")
       
    def testMidpointWeightMatrix(self): 
        '''
        Test that the midpoint weight matrix is generated properly.  Still using 27 point stencil where midpoint = 1 and everything else 0.  
        '''
        W = self.mesh.MidpointWeightMatrix()
        self.assertEqual(np.sum(W), 1.0, "midpoint weights don't sum to 1")    
        self.assertEqual(np.max(W),1.0,"max value not 1.")    
        self.assertEqual(np.min(W),0.0,"max value not 1.")  
        
    def testSimpsonWeightMatrix(self): 
        '''
        Test that the simpson weight matrix is generated properly
        '''
        W = self.mesh.SimpsonWeightMatrix()
        self.assertEqual(np.max(W), 64/27/8, "max not as expected")
        self.assertEqual(np.argmax(W), 13, "max value of W not in center")   
        self.assertEqual(W[0,0,0], W[2,2,2], "corners not equal") 
        self.assertEqual(np.sum(W), 1, "simpson weights dont sum to 1")
        
    def testSimpsonIntegration(self):
        '''
        Test that the simpson cell-based integrator gets quadratic functions exactly
        '''
        def setFakeWavefunction(self):
            '''
            Generate fake wavefunctions used to test the integrators
            '''
            for index in range(len(self.mesh.cells)): 
                    xm,ym,zm = np.meshgrid(self.mesh.cells[index].x,self.mesh.cells[index].y,
                                self.mesh.cells[index].z,indexing='ij')
                    self.mesh.cells[index].psi = xm**2 + ym**2 + zm**2  # quadratic
                
        def evaluateIntegral(self,W):
            I = 0.0
            for index in range(len(self.mesh.cells)):
                I += np.sum( W*self.mesh.cells[index].psi*self.mesh.cells[index].volume )
            return I
            
        self.xmin = self.ymin = self.zmin = -2
        self.xmax = self.ymax = self.zmax =  2
        self.nx = self.ny = self.nz = 8
        self.mesh = Mesh(self.xmin,self.xmax,self.ymin,self.ymax,self.zmin,self.zmax,self.nx,self.ny,self.nz,"analytic")
        
        setFakeWavefunction(self)
#         W = self.mesh.MidpointWeightMatrix()
        W = self.mesh.SimpsonWeightMatrix()
        Integral = evaluateIntegral(self,W)
        self.assertEqual(Integral, 256, "Simpson not integrating quadratic exactly")
        
    def passtestKineticAndPotential(self):  
        '''
        For manually testing the convergence.  Modify the method for refinement, observe the printed errors.  Pass this test unless manually investigating
        '''
        print()
        self.xmin = self.ymin = self.zmin = -8
        self.xmax = self.ymax = self.zmax =  8
        self.nx = self.ny = self.nz = 16
        self.mesh = Mesh(self.xmin,self.xmax,self.ymin,self.ymax,self.zmin,self.zmax,self.nx,self.ny,self.nz,"analytic")
#         W = self.mesh.MidpointWeightMatrix()
        W = self.mesh.SimpsonWeightMatrix()
        psiVariationThreshold = 0.025
        epsilon = 0.025
        levels = 6
        for level in range(levels):
            if level > 0:
                self.mesh.MeshRefinement(psiVariationThreshold)
                self.mesh.setExactWavefunction()
            self.mesh.normalizePsi(W)
            self.mesh.computePotential(W,epsilon)
            self.mesh.computeKinetic(W)
            print('Pass ',level,': ', len(self.mesh.cells),' cells.')
            print('Kinetic error: ', 0.5-self.mesh.Kinetic)
            print('Potential error: ', -1.0-self.mesh.Potential,'\n')
            
    def testKineticAndPotentialImproveWithRefinement(self):  
        '''
        Verify that the potential and kinetic energy calculations get more accurate as mesh is refined.  
        Should hold for reasonable meshes (not too coarse where gradient is meaningless, not too fine that we have midpoints very close to the singular origin)
        '''
        self.xmin = self.ymin = self.zmin = -8
        self.xmax = self.ymax = self.zmax =  8
        self.nx = self.ny = self.nz = 8
        self.mesh = Mesh(self.xmin,self.xmax,self.ymin,self.ymax,self.zmin,self.zmax,self.nx,self.ny,self.nz,"analytic")
        W = self.mesh.MidpointWeightMatrix()
#         W = self.mesh.SimpsonWeightMatrix()
        psiVariationThreshold = 0.1
        epsilon = 0.025
        levels = 3
        coarseKineticError = 0.0
        coarsePotentialError = 0.0
        fineKineticError = 0.0
        finePotentialError = 0.0
        for level in range(levels):
            coarseKineticError = fineKineticError
            coarsePotentialError = finePotentialError
            if level > 0:
                self.mesh.MeshRefinement(psiVariationThreshold)
                self.mesh.setExactWavefunction()
            self.mesh.normalizePsi(W)
            self.mesh.computePotential(W,epsilon)
            self.mesh.computeKinetic(W)
            finePotentialError = abs(-1.0-self.mesh.Potential)
            fineKineticError = abs(0.5-self.mesh.Kinetic)
            if level > 0:
                self.assertLess(finePotentialError, coarsePotentialError, "Potential didn't get more accurate with mesh refinement")
                self.assertLess(fineKineticError, coarseKineticError, "Kinetic didn't get more accurate with mesh refinement")
   
    def testCellConvolve(self):
        '''
        CellConvolve convolves a single target cell with the entire grid of cells.  This tests the accuracy of that convolution for a single target cell,
        particularly its midpoint
        '''
        def index(i,j,k):
            return self.ny*self.nz*i + self.nz*j + k
        targetCell = self.mesh.cells[index(2,2,2)]
        self.mesh.convolution(targetCell)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()