'''
Created on Mar 13, 2018

@author: nathanvaughn
'''
import unittest
import numpy as np
import sys
from timeit import default_timer as timer
sys.path.append('../src/dataStructures')
sys.path.append('../src/utilities')
from TreeStruct import Tree
from convolution import gpuConvolution

class TestGreenIterations(unittest.TestCase):


    @classmethod
    def setUpClass(self):
        '''
        setUp() gets called before every test below.
        '''
        self.xmin = self.ymin = self.zmin = -10
        self.xmax = self.ymax = self.zmax = 10
        self.tree = Tree(self.xmin,self.xmax,self.ymin,self.ymax,self.zmin,self.zmax)
        self.tree.buildTree( minLevels=5, maxLevels=8, divideTolerance=0.075, printTreeProperties=True)
        for element in self.tree.masterList:
            
            element[1].gridpoints[1,1,1].setPsi(np.random.rand(1))
        


    

#     def testGreenIterations(self):
#         self.tree.E = -1.0 # initial guess
#          
#         for i in range(10):
#             print()
#             self.tree.GreenFunctionConvolutionList(timeConvolution=True)
# #             print('Convolution took:                %.4f seconds. ' %self.tree.ConvolutionTime)
#             self.tree.computeWaveErrors()
#             print('Convolution wavefunction errors: %.3e L2,  %.3e max' %(self.tree.L2NormError, self.tree.maxCellError))
#             self.tree.updateEnergy()
#             print('Kinetic Energy:                  %.3f ' %self.tree.totalKinetic)
#             print('Potential Energy:               %.3f ' %self.tree.totalPotential)
#             print('Updated Energy Value:            %.3f Hartree, %.3e error' %(self.tree.E, self.tree.E+0.5))

    def testGreenIterationsGPU(self):
        self.tree.E = -1.0 # initial guess
        
        N = self.tree.numberOfGridpoints
        threadsPerBlock = 512
        blocksPerGrid = (N + (threadsPerBlock - 1)) // threadsPerBlock
        print('Threads per block: ', threadsPerBlock)
        print('Blocks per grid:   ', blocksPerGrid)
        
        GIcounter=1
#         for i in range(20):
        residual = 1
        residualTolerance = 0.00001
        Pold = -10.0
        Kold = -10.0
        while residual > residualTolerance:
            print()
            print('Green Iteration Count ', GIcounter)
            GIcounter+=1
            startExtractionTime = timer()
            sources = self.tree.extractLeavesMidpointsOnly()
            targets = self.tree.extractLeavesAllGridpoints()
            self.assertEqual(self.tree.numberOfGridpoints, len(targets), "targets not equal to number of gridpoints")
            ExtractionTime = timer() - startExtractionTime
            psiNew = np.zeros((len(targets)))
            startConvolutionTime = timer()
            k = np.sqrt(-2*self.tree.E)
#             k = 1
            gpuConvolution[blocksPerGrid, threadsPerBlock](targets,sources,psiNew,k)
            ConvolutionTime = timer() - startConvolutionTime
            print('Extraction took:             %.4f seconds. ' %ExtractionTime)
            print('Convolution took:            %.4f seconds. ' %ConvolutionTime)
#             print('first few values in psiNew:  ', psiNew[100:105])
            self.tree.importPsiOnLeaves(psiNew)
            self.tree.normalizeWavefunction()
            self.tree.computeWaveErrors()
            print('Convolution wavefunction errors: %.3e L2,  %.3e max' %(self.tree.L2NormError, self.tree.maxCellError))
            startEnergyTime = timer()
            self.tree.updateEnergy()
            if self.tree.E > 0.0:
                print('Warning, Energy is positive')
                self.tree.E = -2
            energyUpdateTime = timer() - startEnergyTime
            print('Energy Update took:              %.4f seconds. ' %energyUpdateTime)
            residual = max( abs(Pold - self.tree.totalPotential),abs(Kold - self.tree.totalKinetic) )
            print('Energy Residual:                 %.3e' %residual)
            Pold = self.tree.totalPotential
            Kold = self.tree.totalKinetic
            print('Kinetic Energy Error:            %.5f ' %(self.tree.totalKinetic-0.5))
            print('Potential Energy Error:          %.5f ' %(self.tree.totalPotential+1))
            print('Updated Energy Value:            %.5f Hartree, %.5f error' %(self.tree.E, self.tree.E+0.5))
        print('\nConvergence to a tolerance of %f took %i iterations' %(residualTolerance, GIcounter))
             


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
    
    
    
    
    