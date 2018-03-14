'''
Created on Mar 5, 2018

@author: nathanvaughn
'''
import unittest


from TreeStruct import Tree

class TestEnergyComputation(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.xmin = self.ymin = self.zmin = -10
        self.xmax = self.ymax = self.zmax = -self.xmin
        self.tree = Tree(self.xmin,self.xmax,self.ymin,self.ymax,self.zmin,self.zmax)
        self.tree.buildTree( minLevels=3, maxLevels=7, divideTolerance=0.04, printTreeProperties=True )
        self.tree.normalizeWavefunction()


    def testNormalization(self):
        A = 0.0
        for element in self.tree.masterList:
            if element[1].leaf  == True:
                A += element[1].gridpoints[1,1,1].psi**2*element[1].volume
                
        self.assertAlmostEqual(A, 1.0,12, "wavefunction not normalized")

    def testPotentialComputation(self):
#         self.tree.computePotentialOnTree(epsilon=0, timePotential=True)
#         print('Recursively on tree:')
#         print('\nPotential Error:         %.3g mHartree' %float((-1.0-self.tree.totalPotential)*1000.0))
#         print('Computation took:          %.3g seconds.' %self.tree.PotentialTime)
        
        self.tree.computePotentialOnList(epsilon=0.0, timePotential=True)
#         print('From the master list:')
        print('\nPotential Error:         %.3g mHartree' %float((-1.0-self.tree.totalPotential)*1000.0))
        print('Computation took:          %.3g seconds.' %self.tree.PotentialTime)
#     @unittest.skip("Skip energy computations.")    
    def testKineticComputation(self):
#         self.tree.computeKineticOnTree( timeKinetic=True)
#         print('From tree')
#         print('\nKinetic Error:           %.3g mHartree' %float((0.5-self.tree.totalKinetic)*1000.0))
#         print('Computation took:          %.3g seconds.' %self.tree.KineticTime)
        
        self.tree.computeKineticOnList( timeKinetic=True)
#         print("from master list:")
        print('\nKinetic Error:           %.3g mHartree' %float((0.5-self.tree.totalKinetic)*1000.0))
        print('Computation took:          %.3g seconds.' %self.tree.KineticTime)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()