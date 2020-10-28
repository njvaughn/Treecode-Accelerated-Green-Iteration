<<<<<<< HEAD
'''
Created on Feb 22, 2018

@author: nathanvaughn
'''
import unittest
import numpy as np
import itertools

from CellStruct import Cell
from GridpointStruct import GridPoint
from hydrogenPotential import potential, trueWavefunction

ThreeByThreeByThree = [element for element in itertools.product(range(3),range(3),range(3))]
TwoByTwoByTwo = [element for element in itertools.product(range(2),range(2),range(2))]
FiveByFiveByFive = [element for element in itertools.product(range(5),range(5),range(5))]

class TestCellStructure(unittest.TestCase):
    
    def setUp(self):
        self.xmin = self.ymin = self.zmin = -2
        self.xmax = self.ymax = self.zmax = -1.5
        xvec = np.linspace(self.xmin,self.xmax,3)
        yvec = np.linspace(self.ymin,self.ymax,3)
        zvec = np.linspace(self.zmin,self.zmax,3)
        gridpoints = np.empty((3,3,3),dtype=object)

        for i, j, k in ThreeByThreeByThree:
            gridpoints[i,j,k] = GridPoint(xvec[i],yvec[j],zvec[k])
        
        # generate root cell from the gridpoint objects  
        self.testCell = Cell( gridpoints )


    def tearDown(self):
        self.testCell = None


    def testCellDivide(self):
        '''
        Test cell division.  The original 3x3x3 GridPoint objects should be pointed to by the children
        in addition to the new GridPoint objects created at the refined level.  Check that the gridpoint data 
        gets mapped properly (children are in fact the 8 octants).
        '''
        parent = self.testCell
        parent.uniqueID = '000'
        parent.divide()
        # check that previously existing object are now also owned by the children
        self.assertEqual(parent.gridpoints[2,2,2], parent.children[1,1,1].gridpoints[2,2,2],
                          "corner point not mapped to expected child")
        self.assertEqual(parent.gridpoints[1,1,1], parent.children[1,1,1].gridpoints[0,0,0],
                          "middle point not mapped to expected child")
        self.assertEqual(parent.gridpoints[2,0,1], parent.children[1,0,1].gridpoints[2,0,0],
                          "corner point not mapped to expected child")
        
        # check that children's new cells have the correct new gridpoints
        self.assertEqual(parent.children[0,0,0].gridpoints[1,1,1].x, 
                         (parent.gridpoints[0,0,0].x + parent.gridpoints[1,0,0].x )/2, 
                         "midpoint of child cell not expected value.")
        
        # check that children own same objects on their boundaries
        self.assertEqual(parent.children[0,0,0].gridpoints[1,1,2], parent.children[0,0,1].gridpoints[1,1,0], 
                         "Neighboring children aren't pointing to same gridpoint object on their shared face")

    def testPotentialEvaluation(self):
        volume = 1/8 # size of the setup cell
        psi = trueWavefunction(1, -1.75, -1.75, -1.75)  # wavefunction evaluated at midpoint of cell
        V = potential(-1.75, -1.75, -1.75)  # potential (no smoothing)
        expectedPotential = volume*psi*psi*V
        self.testCell.computePotential()
        self.assertEqual(expectedPotential, self.testCell.PE, "computed potential energy doesn't match expected")
        
    @unittest.skip("Laplacian computation is burried within the kinetic energy computation, not its own method.")
    def testGradientComputation(self):
        def laplacianQuadraticTestFunction(x,y,z):
            return x*x + y*y + z*z  # analytic laplacian = 6
        def laplacianCubicTestFunction(x,y,z):
            return x*x*y + y*y*z + z*x*y  # analytic laplacian = 2*(y+z)
        
        # test quadratic
        for i,j,k in ThreeByThreeByThree:
            gridpt = self.testCell.gridpoints[i,j,k]
            gridpt.setPsi( laplacianQuadraticTestFunction(gridpt.x, gridpt.y, gridpt.z ))
              
        expectedLaplacian = 6
        computedLaplacian = self.testCell.computeLaplacian()
        self.assertEqual(expectedLaplacian, computedLaplacian[1,1,1], "computed kinetic energy doesn't match expected")

        # test cubic
        for i,j,k in ThreeByThreeByThree:
            gridpt = self.testCell.gridpoints[i,j,k]
            gridpt.setPsi( laplacianCubicTestFunction(gridpt.x, gridpt.y, gridpt.z ))
           
        expectedLaplacian = 2*(-1.75-1.75)
        computedLaplacian = self.testCell.computeLaplacian()
        self.assertEqual(expectedLaplacian, computedLaplacian[1,1,1], "computed kinetic energy doesn't match expected")

#     def testKineticEvaluation(self):
#         volume=1/8
#         psi = trueWavefunction(1, -1.75, -1.75, -1.75)
        
        
        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
=======
'''
Created on Feb 22, 2018

@author: nathanvaughn
'''
import unittest
import numpy as np
import itertools

from CellStruct import Cell
from GridpointStruct import GridPoint
from hydrogenPotential import potential, trueWavefunction

ThreeByThreeByThree = [element for element in itertools.product(range(3),range(3),range(3))]
TwoByTwoByTwo = [element for element in itertools.product(range(2),range(2),range(2))]
FiveByFiveByFive = [element for element in itertools.product(range(5),range(5),range(5))]

class TestCellStructure(unittest.TestCase):
    
    def setUp(self):
        self.xmin = self.ymin = self.zmin = -2
        self.xmax = self.ymax = self.zmax = -1.5
        xvec = np.linspace(self.xmin,self.xmax,3)
        yvec = np.linspace(self.ymin,self.ymax,3)
        zvec = np.linspace(self.zmin,self.zmax,3)
        gridpoints = np.empty((3,3,3),dtype=object)

        for i, j, k in ThreeByThreeByThree:
            gridpoints[i,j,k] = GridPoint(xvec[i],yvec[j],zvec[k])
        
        # generate root cell from the gridpoint objects  
        self.testCell = Cell( gridpoints )


    def tearDown(self):
        self.testCell = None


    def testCellDivide(self):
        '''
        Test cell division.  The original 3x3x3 GridPoint objects should be pointed to by the children
        in addition to the new GridPoint objects created at the refined level.  Check that the gridpoint data 
        gets mapped properly (children are in fact the 8 octants).
        '''
        parent = self.testCell
        parent.uniqueID = '000'
        parent.divide()
        # check that previously existing object are now also owned by the children
        self.assertEqual(parent.gridpoints[2,2,2], parent.children[1,1,1].gridpoints[2,2,2],
                          "corner point not mapped to expected child")
        self.assertEqual(parent.gridpoints[1,1,1], parent.children[1,1,1].gridpoints[0,0,0],
                          "middle point not mapped to expected child")
        self.assertEqual(parent.gridpoints[2,0,1], parent.children[1,0,1].gridpoints[2,0,0],
                          "corner point not mapped to expected child")
        
        # check that children's new cells have the correct new gridpoints
        self.assertEqual(parent.children[0,0,0].gridpoints[1,1,1].x, 
                         (parent.gridpoints[0,0,0].x + parent.gridpoints[1,0,0].x )/2, 
                         "midpoint of child cell not expected value.")
        
        # check that children own same objects on their boundaries
        self.assertEqual(parent.children[0,0,0].gridpoints[1,1,2], parent.children[0,0,1].gridpoints[1,1,0], 
                         "Neighboring children aren't pointing to same gridpoint object on their shared face")

    def testPotentialEvaluation(self):
        volume = 1/8 # size of the setup cell
        psi = trueWavefunction(1, -1.75, -1.75, -1.75)  # wavefunction evaluated at midpoint of cell
        V = potential(-1.75, -1.75, -1.75)  # potential (no smoothing)
        expectedPotential = volume*psi*psi*V
        self.testCell.computePotential()
        self.assertEqual(expectedPotential, self.testCell.PE, "computed potential energy doesn't match expected")
        
    @unittest.skip("Laplacian computation is burried within the kinetic energy computation, not its own method.")
    def testGradientComputation(self):
        def laplacianQuadraticTestFunction(x,y,z):
            return x*x + y*y + z*z  # analytic laplacian = 6
        def laplacianCubicTestFunction(x,y,z):
            return x*x*y + y*y*z + z*x*y  # analytic laplacian = 2*(y+z)
        
        # test quadratic
        for i,j,k in ThreeByThreeByThree:
            gridpt = self.testCell.gridpoints[i,j,k]
            gridpt.setPsi( laplacianQuadraticTestFunction(gridpt.x, gridpt.y, gridpt.z ))
              
        expectedLaplacian = 6
        computedLaplacian = self.testCell.computeLaplacian()
        self.assertEqual(expectedLaplacian, computedLaplacian[1,1,1], "computed kinetic energy doesn't match expected")

        # test cubic
        for i,j,k in ThreeByThreeByThree:
            gridpt = self.testCell.gridpoints[i,j,k]
            gridpt.setPsi( laplacianCubicTestFunction(gridpt.x, gridpt.y, gridpt.z ))
           
        expectedLaplacian = 2*(-1.75-1.75)
        computedLaplacian = self.testCell.computeLaplacian()
        self.assertEqual(expectedLaplacian, computedLaplacian[1,1,1], "computed kinetic energy doesn't match expected")

#     def testKineticEvaluation(self):
#         volume=1/8
#         psi = trueWavefunction(1, -1.75, -1.75, -1.75)
        
        
        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
>>>>>>> refs/remotes/eclipse_auto/master
    unittest.main()