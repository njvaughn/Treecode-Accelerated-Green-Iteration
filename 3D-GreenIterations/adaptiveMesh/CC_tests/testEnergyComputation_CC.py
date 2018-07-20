'''
Created on Mar 5, 2018

@author: nathanvaughn
'''
import unittest
import sys
sys.path.append('../src/dataStructures')
sys.path.append('../src/utilities')
import itertools
ThreeByThreeByThree = [element for element in itertools.product(range(3),range(3),range(3))]

from TreeStruct_CC import Tree

class TestEnergyComputation(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        print('Building tree...')
        self.xmin = self.ymin = self.zmin = -10
        self.xmax = self.ymax = self.zmax = -self.xmin
        self.px = self.py = self.pz = 5
        self.tree = Tree(self.xmin,self.xmax,self.px,
                         self.ymin,self.ymax,self.py,
                         self.zmin,self.zmax,self.pz)
#         self.tree.buildTree( minLevels=5, maxLevels=20, N=1000, printTreeProperties=True )
        self.tree.buildTree( minLevels=2, maxLevels=15, divideCriterion='parent_child_PE', divideParameter=1e-5, printTreeProperties=True)

#         self.tree.buildTree( minLevels=1, maxLevels=12, maxDx=0.75, divideTolerance1=0.05, divideTolerance2=5e-6, printTreeProperties=True )
#         self.tree.buildTreeOneCondition( minLevels=4, maxLevels=12, divideTolerance=0.04, printTreeProperties=True )
#         self.tree.normalizeWavefunction()
        

    def testEnergy(self):
        print('Order: %i' %self.px)
        print('\nComputing Ground State energy...')
        # set wavefunction to FES, repeat energy calculation
        self.tree.populatePsiWithAnalytic(0)
#         for element in self.tree.masterList:
#             for i,j,k in self.tree.PxByPyByPz:
#                 element[1].gridpoints[i,j,k].setAnalyticPsi(0)
#         self.tree.normalizeWavefunction()
         
#         self.tree.computeKineticOnList()
#         self.tree.computePotentialOnList(epsilon=0.0)
# #         print('\nBefore Normalization:')
#         print(  'Ground State Energy:            %.6g Hartree' %float((self.tree.totalKinetic+self.tree.totalPotential)))
#         print(  'Potential Energy Error:         %.6g mHartree' %float((-1.0-self.tree.totalPotential)*1000.0))
#         print(  'Kinetic Energy Error:           %.6g mHartree' %float((0.5-self.tree.totalKinetic)*1000.0))
#         print(  'Ground State Error:             %.6g mHartree' %float((-0.5-self.tree.totalKinetic-self.tree.totalPotential)*1000.0))
 
 
        self.tree.normalizeWavefunction()
         
        self.tree.computeKineticOnList()
        self.tree.computePotentialOnList(epsilon=0.0)
#         print('\n\nAfter Normalization:')
        print(  'Ground State Energy:            %.6g Hartree' %float((self.tree.totalKinetic+self.tree.totalPotential)))
        print(  'Potential Energy Error:         %.6g mHartree' %float((-1.0-self.tree.totalPotential)*1000.0))
        print(  'Kinetic Energy Error:           %.6g mHartree' %float((0.5-self.tree.totalKinetic)*1000.0))
        print(  'Ground State Error:             %.6g mHartree' %float((-0.5-self.tree.totalKinetic-self.tree.totalPotential)*1000.0))
 
#         print('\nComputing First Excited State energy...')
#         # set wavefunction to FES, repeat energy calculation
#         for element in self.tree.masterList:
#             for i,j,k in ThreeByThreeByThree:
#                 element[1].gridpoints[i,j,k].setAnalyticPsi(1)
# #         self.tree.normalizeWavefunction()
#           
#         self.tree.computeKineticOnList()
#         self.tree.computePotentialOnList(epsilon=0.0)       
#         print('\nExcited State Energy:           %.6g Hartree' %float((self.tree.totalKinetic+self.tree.totalPotential)))
#         print(  'Potential Energy Error:         %.6g mHartree' %float((-0.25-self.tree.totalPotential)*1000.0))
#         print(  'Kinetic Energy Error:           %.6g mHartree' %float((0.125-self.tree.totalKinetic)*1000.0))
#         print(  'Excited State Error:            %.6g mHartree' %float((-0.125-self.tree.totalKinetic-self.tree.totalPotential)*1000.0))



#     def testPotentialComputation(self):
# #         self.tree.computePotentialOnTree(epsilon=0, timePotential=True)
# #         print('Recursively on tree:')
# #         print('\nPotential Error:         %.3g mHartree' %float((-1.0-self.tree.totalPotential)*1000.0))
# #         print('Computation took:          %.3g seconds.' %self.tree.PotentialTime)
#          
#         self.tree.computePotentialOnList(epsilon=0.0, timePotential=True)
# #         print('From the master list:')
#         print('\nPotential Error:         %.3g mHartree' %float((-1.0-self.tree.totalPotential)*1000.0))
#         print('Computation took:          %.3g seconds.' %self.tree.PotentialTime)
# #     @unittest.skip("Skip energy computations.")    
#     def testKineticComputation(self):
# #         self.tree.computeKineticOnTree( timeKinetic=True)
# #         print('From tree')
# #         print('\nKinetic Error:           %.3g mHartree' %float((0.5-self.tree.totalKinetic)*1000.0))
# #         print('Computation took:          %.3g seconds.' %self.tree.KineticTime)
#          
#         self.tree.computeKineticOnList( timeKinetic=True)
# #         print("from master list:")
#         print('\nKinetic Error:           %.3g mHartree' %float((0.5-self.tree.totalKinetic)*1000.0))
#         print('Computation took:          %.3g seconds.' %self.tree.KineticTime)
        

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()