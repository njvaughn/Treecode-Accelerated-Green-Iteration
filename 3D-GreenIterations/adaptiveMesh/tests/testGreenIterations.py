'''
testGreenIterations.py
This is a unitTest module for testing Green iterations.  It begins by building the tree-based
adaotively refined mesh, then performs Green iterations to obtain the ground state energy
and wavefunction for the single electron hydrogen atom.  -- 03/20/2018 NV

Created on Mar 13, 2018
@author: nathanvaughn
'''

import sys
sys.path.append('../src/dataStructures')
sys.path.append('../src/utilities')

import unittest
import numpy as np
from timeit import default_timer as timer

from TreeStruct import Tree
from convolution import gpuConvolution

class TestGreenIterations(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        '''
        setUp() gets called before every test below.
        '''
        self.xmin = self.ymin = self.zmin = -8
        self.xmax = self.ymax = self.zmax = 8
        self.tree = Tree(self.xmin,self.xmax,self.ymin,self.ymax,self.zmin,self.zmax)
        self.tree.buildTree( minLevels=4, maxLevels=9, divideTolerance=0.05, printTreeProperties=True)
        for element in self.tree.masterList:
            
            element[1].gridpoints[1,1,1].setPsi(np.random.rand(1))
        


    
    @unittest.skip('Skipped CPU convolution')    
    def testGreenIterations(self):
        self.tree.E = -1.0 # initial guess
          
        for i in range(1):
            print()
            self.tree.GreenFunctionConvolutionList(timeConvolution=True)
            print('Convolution took:                %.4f seconds. ' %self.tree.ConvolutionTime)
            self.tree.computeWaveErrors()
            print('Convolution wavefunction errors: %.3e L2,  %.3e max' %(self.tree.L2NormError, self.tree.maxCellError))
            self.tree.updateEnergy()
            print('Kinetic Energy:                  %.3f ' %self.tree.totalKinetic)
            print('Potential Energy:               %.3f ' %self.tree.totalPotential)
            print('Updated Energy Value:            %.3f Hartree, %.3e error' %(self.tree.E, self.tree.E+0.5))

    def testGreenIterationsGPU(self):
        self.tree.E = -1.0 # set initial energy guess
        
        N = self.tree.numberOfGridpoints                # set N to be the number of gridpoints.  These will be all the targets
        threadsPerBlock = 512                           # set the number of threads per block.  512 seems to perform well, but can also try larger or smaller.  Should be a multiple of 32 since that is the warp size.
        blocksPerGrid = (N + (threadsPerBlock - 1)) // threadsPerBlock  # compute the number of blocks based on N and threadsPerBlock
        print('Threads per block: ', threadsPerBlock)
        print('Blocks per grid:   ', blocksPerGrid)
        
        
        GIcounter=1                                     # initialize the counter to counter the number of iterations required for convergence
        residual = 1                                    # initialize the residual to something that fails the convergence tolerance
        residualTolerance = 0.000005                      # set the residual tolerance
        Eold = -10.0
#         Pold = -10.0                                    # set initial values of potential and kinetic energy, these will be overwritten once the iterations begin.
#         Kold = -10.0
        while residual > residualTolerance:
            print()
            print('Green Iteration Count ', GIcounter)
            GIcounter+=1
            startExtractionTime = timer()
            sources = self.tree.extractLeavesMidpointsOnly()  # extract the source point locations.  Currently, these are just all the leaf midpoints
            targets = self.tree.extractLeavesAllGridpoints()  # extract the target point locations.  Currently, these are all 27 gridpoints per cell (no redundancy)
            self.assertEqual(self.tree.numberOfGridpoints, len(targets), "targets not equal to number of gridpoints") # verify that the number of targets equals the number of total gridpoints of the tree
            ExtractionTime = timer() - startExtractionTime
            psiNew = np.zeros((len(targets)))
            startConvolutionTime = timer()
            k = np.sqrt(-2*self.tree.E)                 # set the k value coming from the current guess for the energy
#             k = 1
            gpuConvolution[blocksPerGrid, threadsPerBlock](targets,sources,psiNew,k)  # call the GPU convolution
            ConvolutionTime = timer() - startConvolutionTime
            print('Extraction took:             %.4f seconds. ' %ExtractionTime)
            print('Convolution took:            %.4f seconds. ' %ConvolutionTime)

            self.tree.importPsiOnLeaves(psiNew)         # import the new wavefunction values into the tree.
            self.tree.normalizeWavefunction()           # Normalize the new wavefunction 
            self.tree.computeWaveErrors()               # Compute the wavefunction errors compared to the analytic ground state 
            print('Convolution wavefunction errors: %.3e L2,  %.3e max' %(self.tree.L2NormError, self.tree.maxCellError))
            startEnergyTime = timer()
            self.tree.updateEnergy()                    # Compute the new energy values
            if self.tree.E > 0.0:                       # Check that the current guess for energy didn't go positive.  Reset it if it did. 
                print('Warning, Energy is positive')
                self.tree.E = -2
            energyUpdateTime = timer() - startEnergyTime
            print('Energy Update took:              %.4f seconds. ' %energyUpdateTime)
#             residual = max( abs(Pold - self.tree.totalPotential),abs(Kold - self.tree.totalKinetic) )  # Compute the residual for determining convergence
            residual = abs(Eold - self.tree.E)  # Compute the residual for determining convergence
            print('Energy Residual:                 %.3e' %residual)
#             Pold = self.tree.totalPotential             # Update the old Potential and Kinetic 
#             Kold = self.tree.totalKinetic
            Eold = self.tree.E
            print('Kinetic Energy Error:            %.5f ' %(self.tree.totalKinetic-0.5))
            print('Potential Energy Error:          %.5f ' %(self.tree.totalPotential+1))
            print('Updated Energy Value:            %.5f Hartree, %.6f error' %(self.tree.E, self.tree.E+0.5))
        print('\nConvergence to a tolerance of %f took %i iterations' %(residualTolerance, GIcounter))
             


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
    
    
    
    
    