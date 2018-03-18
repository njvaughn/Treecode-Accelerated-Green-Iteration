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
        self.tree.buildTree( minLevels=3, maxLevels=3, divideTolerance=0.04, printTreeProperties=True)
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
        
        for i in range(10):
            print()
            startExtractionTime = timer()
            sources = self.tree.extractLeavesMidpointsOnly()
            targets = self.tree.extractLeavesAllGridpoints()
            ExtractionTime = timer() - startExtractionTime
            psiNew = np.zeros((len(targets)))
            startConvolutionTime = timer()
            gpuConvolution(targets,sources,psiNew,self.tree.E)
            ConvolutionTime = timer() - startConvolutionTime
            print('Extraction took:             %.4f seconds. ' %ExtractionTime)
            print('Convolution took:            %.4f seconds. ' %ConvolutionTime)

            self.tree.importPsiOnLeaves(psiNew)
            self.tree.computeWaveErrors()
            print('Convolution wavefunction errors: %.3e L2,  %.3e max' %(self.tree.L2NormError, self.tree.maxCellError))
            startEnergyTime = timer()
            self.tree.updateEnergy()
            energyUpdateTime = timer() - startEnergyTime
            print('Energy Update took:          %.4f seconds. ' %energyUpdateTime)

            print('Kinetic Energy:                  %.3f ' %self.tree.totalKinetic)
            print('Potential Energy:               %.3f ' %self.tree.totalPotential)
            print('Updated Energy Value:            %.3f Hartree, %.3e error' %(self.tree.E, self.tree.E+0.5))
            


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
    
    
    
    
    