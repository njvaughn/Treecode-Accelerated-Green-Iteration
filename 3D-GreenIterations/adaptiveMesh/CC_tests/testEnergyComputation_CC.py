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
        inputFile ='../src/utilities/molecularConfigurations/carbonMonoxideAuxiliary.csv'
#         inputFile ='../src/utilities/molecularConfigurations/carbonAtomAuxiliary.csv'
        xmin = ymin = zmin = -20
        xmax = ymax = zmax = 20
        order=3
        minDepth=3
        maxDepth=15
        divideCriterion='LW3'
        divideParameter=800
        
        [coordinateFile, outputFile] = np.genfromtxt(inputFile,dtype="|U100")[:2]
        [nElectrons, nOrbitals, Etrue, ExTrue, EcTrue, Eband, gaugeShift] = np.genfromtxt(inputFile)[2:]
        nElectrons = int(nElectrons)
        nOrbitals = int(nOrbitals)
        
        print([coordinateFile, outputFile, nElectrons, nOrbitals, 
         Etrue, ExTrue, EcTrue, Eband, gaugeShift])
        self.tree = Tree(xmin,xmax,order,ymin,ymax,order,zmin,zmax,order,nElectrons,nOrbitals,gaugeShift=gaugeShift,
                    coordinateFile=coordinateFile,inputFile=inputFile)#, iterationOutFile=outputFile)
    
        self.tree.buildTree( minLevels=minDepth, maxLevels=maxDepth, initializationType='atomic',divideCriterion=divideCriterion, divideParameter=divideParameter, printTreeProperties=True)
#         self.tree.orthonormalizeOrbitals()
        self.tree.occupations = np.array([2,2,1/3,1/3,1/3,2,2,2/3,2/3,2/3])

    def testEnergyComputation(self):
        start = time.time()
        self.tree.computeDerivativeMatrices()
        Dmatrices = time.time()-start
        print('Time to compute derivative matrices: ', Dmatrices)
        start=time.time()
        self.tree.updateOrbitalEnergies(sortByEnergy=False)
        end = time.time()
        print('Time to compute energy: ', end-start)
        print()
        print(self.tree.orbitalEnergies)
            
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()