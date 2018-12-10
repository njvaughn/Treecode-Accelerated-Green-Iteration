'''
Created on Mar 5, 2018

@author: nathanvaughn
'''
import unittest
import sys
import numpy as np
import time
sys.path.append('../src/dataStructures')
sys.path.append('../src/utilities')
import itertools
ThreeByThreeByThree = [element for element in itertools.product(range(3),range(3),range(3))]

from TreeStruct_CC import Tree

class TestEnergyComputation(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        '''
        setUp() gets called before every test below.
        '''
#         inputFile ='../src/utilities/molecularConfigurations/berylliumAuxiliary.csv'
#         inputFile ='../src/utilities/molecularConfigurations/carbonMonoxideAuxiliary.csv'
        inputFile ='../src/utilities/molecularConfigurations/carbonAtomAuxiliary.csv'
        xmin = ymin = zmin = -20
        xmax = ymax = zmax = 20
        order=3
        minDepth=3
        maxDepth=15
        divideCriterion='LW5'
        divideParameter=200
        
        [coordinateFile, DummyOutputFile] = np.genfromtxt(inputFile,dtype="|U100")[:2]
        [nElectrons, nOrbitals, Eband, Ekinetic, Eexchange, Ecorrelation, Eelectrostatic, Etotal, gaugeShift] = np.genfromtxt(inputFile)[2:]
        nElectrons = int(nElectrons)
        nOrbitals = int(nOrbitals)
    
        tree = Tree(xmin,xmax,order,ymin,ymax,order,zmin,zmax,order,nElectrons,nOrbitals,maxDepthAtAtoms=maxDepth,gaugeShift=gaugeShift,
                coordinateFile=coordinateFile,inputFile=inputFile)#, iterationOutFile=outputFile)
        self.tree = tree
    
    
        print('max depth ', maxDepth)
        self.tree.buildTree( minLevels=minDepth, maxLevels=maxDepth, initializationType='random',divideCriterion=divideCriterion, divideParameter=divideParameter, printTreeProperties=True,onlyFillOne=False)
    #     for element in tree.masterList:

#     def testEnergyComputation(self):
#         start = time.time()
#         self.tree.computeDerivativeMatrices()
#         Dmatrices = time.time()-start
#         print('Time to compute derivative matrices: ', Dmatrices)
#         start=time.time()
#         self.tree.updateOrbitalEnergies(sortByEnergy=False)
#         end = time.time()
#         print('Time to compute energy: ', end-start)
#         print()
#         print(self.tree.orbitalEnergies)

    def testExportWeights(self):
        
        densityExport = self.tree.extractLeavesDensity()
        weights_density = densityExport[:,4]
        
        psi0Export = self.tree.extractPhi(0)
        psi1Export = self.tree.extractPhi(1)
        
        weight_psi0 = psi0Export[:,5]
        weight_psi1 = psi1Export[:,5]
        
        
        print(np.max(np.abs(weights_density-weight_psi0)))
        print(np.max(np.abs(weight_psi1-weight_psi0)))
        print(self.tree.root.PxByPyByPz)
        print(self.tree.root.children[0,1,1].PxByPyByPz)
        
            
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()