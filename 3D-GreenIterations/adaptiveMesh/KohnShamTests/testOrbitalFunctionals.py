'''
Created on Jul 9, 2018

@author: nathanvaughn
'''
import unittest
import numpy as np
from TreeStruct_CC import Tree


class Test(unittest.TestCase):

    @classmethod
    def setUp(self):
        xmin = ymin = zmin = -10
        xmax = ymax = zmax = 10
        order = 3
        coordinateFile = '../src/utilities/molecularConfigurations/hydrogenAtom.csv'
        self.tree = Tree(xmin,xmax,order,ymin,ymax,order,zmin,zmax,order,coordinateFile)
    
        self.tree.buildTree( minLevels=2, maxLevels=10, divideCriterion='LW1', divideParameter=150, printTreeProperties=True)

        for _,cell in self.tree.masterList:
            for i,j,k in cell.PxByPyByPz:
                gp = cell.gridpoints[i,j,k]
                r = np.sqrt(gp.x*gp.x + gp.y*gp.y + gp.z*gp.z)
                gp.phi = np.exp(-r)
        self.tree.normalizeOrbital()
        

    def tearDown(self):
        self.tree = None


    def testOrbitalKineticEnergy(self):
        self.tree.computeOrbitalKinetic()
        print('Orbital kinetic energy: ', self.tree.orbitalKinetic)
    
    def testOrbitalPotentialEnergy(self):
        self.tree.computeOrbitalPotential()
        print('Orbital potential energy: ', self.tree.orbitalPotential)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()