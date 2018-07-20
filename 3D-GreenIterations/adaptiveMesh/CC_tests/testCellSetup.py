'''
Created on June 7, 2018

@author: nathanvaughn
'''
import unittest
import numpy as np
from numpy.random import randint
import itertools

from CellStruct_CC import Cell
from GridpointStruct import GridPoint
from hydrogenPotential import potential, trueWavefunction
from meshUtilities import ChebyshevPoints

ThreeByThreeByThree = [element for element in itertools.product(range(3),range(3),range(3))]
TwoByTwoByTwo = [element for element in itertools.product(range(2),range(2),range(2))]
FiveByFiveByFive = [element for element in itertools.product(range(5),range(5),range(5))]

class TestCellStructure(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.xmin = self.ymin = self.zmin = -2
        self.xmax = self.ymax = self.zmax = 1.5
        self.order = 5
        
        self.px = self.order
        self.py = self.order
        self.pz = self.order
        
        xvec = ChebyshevPoints(self.xmin,self.xmax,self.px)
        yvec = ChebyshevPoints(self.ymin,self.ymax,self.py)
        zvec = ChebyshevPoints(self.zmin,self.zmax,self.pz)
        gridpoints = np.empty((self.px,self.py,self.pz),dtype=object)

        PxByPybyPz = [element for element in itertools.product(range(self.px),range(self.py),range(self.pz))]
        for i, j, k in PxByPybyPz:
            gridpoints[i,j,k] = GridPoint(xvec[i],yvec[j],zvec[k])
        
        # generate root cell from the gridpoint objects  
        self.testCell = Cell(self.xmin, self.xmax, self.px, 
                             self.ymin, self.ymax, self.py, 
                             self.zmin, self.zmax, self.pz, gridpoints )


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
        parent.divide(xdiv=parent.xmid, ydiv=parent.ymid, zdiv=parent.zmid)
        
        self.assertEqual(parent.xmid,parent.children[0,randint(2),randint(2)].xmax, 
                         "Parent's x midpoint should be right end of left children")
        self.assertEqual(parent.xmid,parent.children[1,randint(2),randint(2)].xmin, 
                         "Parent's x midpoint should be left end of right children")
        self.assertEqual(parent.ymid,parent.children[randint(2),0,randint(2)].ymax, 
                         "Parent's y midpoint should be right end of left children")
        self.assertEqual(parent.ymid,parent.children[randint(2),1,randint(2)].ymin, 
                         "Parent's y midpoint should be left end of right children")
        self.assertEqual(parent.zmid,parent.children[randint(2),randint(2),0].zmax, 
                         "Parent's z midpoint should be right end of left children")
        self.assertEqual(parent.zmid,parent.children[randint(2),randint(2),1].zmin, 
                         "Parent's z midpoint should be left end of right children")
    
    def testCellDivideInto4(self):
        '''
        Test cell division.  The original 3x3x3 GridPoint objects should be pointed to by the children
        in addition to the new GridPoint objects created at the refined level.  Check that the gridpoint data 
        gets mapped properly (children are in fact the 8 octants).
        '''
        parent = self.testCell
        parent.uniqueID = '000'
        parent.divideInto4(xdiv=0.2, ydiv=0.2, zdiv=None )
        for i in range(2):
            for j in range(2):
                print('Child (%i, %i)'%(i,j))
                print('UniqueID ', parent.children[i,j,0].uniqueID)
        
        self.assertEqual(np.shape(parent.children), (2,2,1), "Children not right shape")
        
        # test z axis
        self.assertEqual(parent.zmin,parent.children[randint(2),randint(2),0].zmin, 
                         "Children have same zmin as parent")
        self.assertEqual(parent.zmid,parent.children[randint(2),randint(2),0].zmid, 
                         "Children have same zmid as parent")
        self.assertEqual(parent.zmax,parent.children[randint(2),randint(2),0].zmax, 
                         "Children have same zmax as parent")
        
        parent.children=None
        parent.divideInto4(xdiv=0.2, ydiv=None, zdiv=0.1 )
        for i in range(2):
            for k in range(2):
                print('Child (%i, %i)'%(i,j))
                print('UniqueID ', parent.children[i,0,k].uniqueID)
        
        self.assertEqual(np.shape(parent.children), (2,1,2), "Children not right shape")
        
        # test y axis
        self.assertEqual(parent.ymin,parent.children[randint(2),0,randint(2)].ymin, 
                         "Children don't have same ymin as parent")
        self.assertEqual(parent.ymid,parent.children[randint(2),0,randint(2)].ymid, 
                         "Children don't have same ymid as parent")
        self.assertEqual(parent.ymax,parent.children[randint(2),0,randint(2)].ymax, 
                         "Children don't have same ymax as parent")
        
        parent.children=None
        parent.divideInto4(xdiv=None, ydiv=0.22, zdiv=0.1 )
        for j in range(2):
            for k in range(2):
                print('Child (%i, %i)'%(i,j))
                print('UniqueID ', parent.children[0,j,k].uniqueID)
        
        self.assertEqual(np.shape(parent.children), (1,2,2), "Children not right shape")
        
        # test x axis
        self.assertEqual(parent.xmin,parent.children[0,randint(2),randint(2)].xmin, 
                         "Children don't have same ymin as parent")
        self.assertEqual(parent.xmid,parent.children[0,randint(2),randint(2)].xmid, 
                         "Children don't have same ymid as parent")
        self.assertEqual(parent.xmax,parent.children[0,randint(2),randint(2)].xmax, 
                         "Children don't have same ymax as parent")
        
        parent.divideInto4(xdiv=None, ydiv=None, zdiv=0.1 )
        
 
    def testCellOrder(self):
        self.assertEqual(self.order, self.testCell.px, "Cell order doesn't match input order.")
        self.assertEqual(np.shape(self.testCell.gridpoints), (self.order,self.order,self.order), 
                         "Dimension of gridpoints not correct")
        self.assertEqual(np.shape(self.testCell.gridpoints), (self.testCell.px,self.testCell.py,self.testCell.pz), 
                         "Dimension of gridpoints not correct")
        
    def testGridpoints(self):
        i = np.random.randint(self.px)
        j = np.random.randint(self.py)
        k = np.random.randint(self.pz)
        testGP = self.testCell.gridpoints[i,j,k]
        self.assertEqual( testGP.x, ChebyshevPoints(self.xmin,self.xmax,self.px)[i], "x value not correct" )
        self.assertEqual( testGP.y, ChebyshevPoints(self.ymin,self.ymax,self.py)[j], "y value not correct" )
        self.assertEqual( testGP.z, ChebyshevPoints(self.zmin,self.zmax,self.pz)[k], "z value not correct" )
        
    @unittest.skip("Potential evaluation still using midpoint method")
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

    unittest.main()