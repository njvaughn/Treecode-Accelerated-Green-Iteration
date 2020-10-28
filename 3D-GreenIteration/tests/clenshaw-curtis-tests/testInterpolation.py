'''
Created on Jul 20, 2018

@author: natha
'''
import unittest
import numpy as np
import itertools

from CellStruct_CC import Cell
from GridpointStruct import GridPoint
from meshUtilities import ChebyshevPoints


def polynomial(x,y,z):
    return x**2 - y**2 + z**3

def exponential(x,y,z):
    return np.exp(-x**2 - y**2 - z**2)

def sinusoidal(x,y,z):
    return np.sin(x)*np.cos(y)*np.cos(z)

class TestInterpolation(unittest.TestCase):


    @classmethod
    def setUpClass(self):
        
        
        self.xmin = self.ymin = self.zmin = -1
        self.xmax = self.ymax = self.zmax = 1
        self.order = 7
        
        self.px = self.order
        self.py = self.order
        self.pz = self.order
        
        self.nOrbitals=3
        
        xvec = ChebyshevPoints(self.xmin,self.xmax,self.px)
        yvec = ChebyshevPoints(self.ymin,self.ymax,self.py)
        zvec = ChebyshevPoints(self.zmin,self.zmax,self.pz)
        
        self.xvec = xvec
        self.yvec = yvec
        self.zvec = zvec
        
        gridpoints = np.empty((self.px,self.py,self.pz),dtype=object)

        PxByPybyPz = [element for element in itertools.product(range(self.px),range(self.py),range(self.pz))]
        for i, j, k in PxByPybyPz:
            gridpoints[i,j,k] = GridPoint(xvec[i],yvec[j],zvec[k],self.nOrbitals)
            poly = polynomial(xvec[i],yvec[j],zvec[k])
            exp = exponential(xvec[i],yvec[j],zvec[k])
            sinu = sinusoidal(xvec[i],yvec[j],zvec[k])
            gridpoints[i,j,k].setPhi(poly,0)  # set the first orbital to the polnomial
            gridpoints[i,j,k].setPhi(exp,1)   # set the second to the exponential
            gridpoints[i,j,k].setPhi(sinu,2)   # set the second to the exponential
        
        # generate root cell from the gridpoint objects  
        self.testCell = Cell(self.xmin, self.xmax, self.px, 
                             self.ymin, self.ymax, self.py, 
                             self.zmin, self.zmax, self.pz, gridpoints )
        


    def tearDown(self):
        self.testCell=None
        


    def testInterpolator(self):
        poly = np.zeros((self.px,self.py,self.pz))
        expo = np.zeros((self.px,self.py,self.pz))
        sinu = np.zeros((self.px,self.py,self.pz))
        for i,j,k in self.testCell.PxByPyByPz:
            poly[i,j,k] = self.testCell.gridpoints[i,j,k].phi[0]
            expo[i,j,k] = self.testCell.gridpoints[i,j,k].phi[1]
            sinu[i,j,k] = self.testCell.gridpoints[i,j,k].phi[2]
        
        i,j,k = np.random.randint(0,self.order,3)
        self.assertEqual(poly[i,j,k], polynomial(self.xvec[i],self.yvec[j],self.zvec[k]), "polnomial value not correct to begin with")
        self.assertEqual(expo[i,j,k], exponential(self.xvec[i],self.yvec[j],self.zvec[k]), "exponential value not correct to begin with")
        self.assertEqual(sinu[i,j,k], sinusoidal(self.xvec[i],self.yvec[j],self.zvec[k]), "sinusoidal value not correct to begin with")
        
        PolyInt = self.testCell.interpolator(self.xvec,self.yvec,self.zvec,poly)
        ExpoInt = self.testCell.interpolator(self.xvec,self.yvec,self.zvec,expo)
        SinuInt = self.testCell.interpolator(self.xvec,self.yvec,self.zvec,sinu)
        
        xt,yt,zt = 2*np.random.rand(3)-1
        
        self.assertAlmostEqual(polynomial(xt,yt,zt), PolyInt(xt,yt,zt), 12, "Polynomial interpolator not accurate")
        self.assertAlmostEqual(exponential(xt,yt,zt), ExpoInt(xt,yt,zt), 3, "Exponential interpolator not accurate")
        self.assertAlmostEqual(sinusoidal(xt,yt,zt), SinuInt(xt,yt,zt), 3, "Sinusoidal interpolator not accurate")
        
        xt,yt,zt = 2*np.random.rand(3)-1
        
        self.assertAlmostEqual(polynomial(xt,yt,zt), PolyInt(xt,yt,zt), 12, "Polynomial interpolator not accurate")
        self.assertAlmostEqual(exponential(xt,yt,zt), ExpoInt(xt,yt,zt), 3, "Exponential interpolator not accurate")
        self.assertAlmostEqual(sinusoidal(xt,yt,zt), SinuInt(xt,yt,zt), 4, "Sinusoidal interpolator not accurate")
        
        
    
    def testInterpolationDuringDivision(self):
        poly = np.zeros((self.px,self.py,self.pz))
        expo = np.zeros((self.px,self.py,self.pz))
        sinu = np.zeros((self.px,self.py,self.pz))
        for i,j,k in self.testCell.PxByPyByPz:
            poly[i,j,k] = self.testCell.gridpoints[i,j,k].phi[0]
            expo[i,j,k] = self.testCell.gridpoints[i,j,k].phi[1]
            sinu[i,j,k] = self.testCell.gridpoints[i,j,k].phi[2]
            
        PolyInt = self.testCell.interpolator(self.xvec,self.yvec,self.zvec,poly)
        ExpoInt = self.testCell.interpolator(self.xvec,self.yvec,self.zvec,expo)
        SinuInt = self.testCell.interpolator(self.xvec,self.yvec,self.zvec,sinu)
        
        
        self.testCell.divide(xdiv=0,ydiv=0,zdiv=0,interpolate=True)
        
        ic,jc,kc = np.random.randint(0,2,3)
        childCell = self.testCell.children[ic,jc,kc]
        
        i,j,k = np.random.randint(0,self.order,3)
        testPoint = childCell.gridpoints[i,j,k]
        
        self.assertEqual(testPoint.phi[0], PolyInt( testPoint.x, testPoint.y, testPoint.z), "Child's gridpoint didn't receive interpolated phi[0]")
        self.assertEqual(testPoint.phi[1], ExpoInt( testPoint.x, testPoint.y, testPoint.z), "Child's gridpoint didn't receive interpolated phi[1]")
        self.assertEqual(testPoint.phi[2], SinuInt( testPoint.x, testPoint.y, testPoint.z), "Child's gridpoint didn't receive interpolated phi[2]")


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()