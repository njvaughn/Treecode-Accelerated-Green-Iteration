'''
Created on June 7, 2018

@author: nathanvaughn
'''
import unittest
import numpy as np
from numpy.random import randint
import itertools

import sys
sys.path.append('../src/dataStructures')
sys.path.append('../src/utilities')

from meshUtilities import *
from GridpointStruct import *

class TestQuadratureFunctions(unittest.TestCase):
    
    def setUp(self):
        self.xmin = self.ymin = self.zmin = -2
        self.xmax = self.ymax = self.zmax = -1.5
        self.order = 4
        
        self.px = self.order
        self.py = self.order
        self.pz = self.order
        
        xvec = ChebyshevPointsFirstKind(self.xmin,self.xmax,self.px)
        yvec = ChebyshevPointsFirstKind(self.ymin,self.ymax,self.py)
        zvec = ChebyshevPointsFirstKind(self.zmin,self.zmax,self.pz)
        gridpoints = np.empty((self.px,self.py,self.pz),dtype=object)

        PxByPybyPz = [element for element in itertools.product(range(self.px),range(self.py),range(self.pz))]
#         for i, j, k in PxByPybyPz:
#             gridpoints[i,j,k] = GridPoint(xvec[i],yvec[j],zvec[k])
        



    def testWeightVector(self):
        W = unscaledWeightsFirstKind(5)
        w1 = weightsFirstKind(-1,2,5)
        w2 = weightsFirstKind(-1,2,5,w=W)
        self.assertTrue( np.alltrue(w1 == w2) , "Weights not equal")
        self.assertAlmostEqual(np.sum(w1),(2+1),14, "Weights do not sum to interval length")
    
    def testWeightMatrix(self):
        W = unscaledWeightsFirstKind(5)
        w3 = weights3DFirstKind(-1, 1, 5, -1, 1, 5, -1, 1, 5, W)
        self.assertEqual(np.shape(w3), (6,6,6), "3D weight matrix not right shape.")
        
    def testOpenChebyshevPoints(self):
        x = ChebyshevPointsFirstKind(-1, 1, 5)
        endpoints = np.linspace(np.pi,0,5+2)
        theta = (endpoints[1:] + endpoints[:-1])/2
        self.assertTrue( np.allclose( theta, np.arccos(x), rtol=1.e-12, atol=1.e-12,),
                         "arccos(x) not almost equal to theta" )
        self.assertGreater(x[0], -1, "Left endpoint not open")
        self.assertLess(x[-1], 1, "Right endpoint not open")
        self.assertEqual(len(x), 6, "x not correct length")
        
    def testIntegration(self):
        p = 5
        xvec = ChebyshevPointsFirstKind(0,1,p)
        yvec = ChebyshevPointsFirstKind(0,1,p)
        zvec = ChebyshevPointsFirstKind(0,1,p)
        W = unscaledWeightsFirstKind(p)
        
        xm,ym,zm = np.meshgrid(xvec,yvec,zvec)
        fm = xm**2 + ym**3 + zm**4
        w = weights3DFirstKind(0, 1, p, 0, 1, p, 0, 1, p, W)
        I_cc = np.sum(fm*w)
        I_anal = 1/3+1/4+1/5
        self.assertAlmostEqual(I_cc, I_anal, 14, "Clenshaw-Curtis didn't integrate 3D polynomial accurately.")
        
    def testChebGradient3D(self):
        print('Testing Cheb Gradients...')
        p = 3
        xvec = ChebyshevPointsFirstKind(0,0.1,p)
        yvec = ChebyshevPointsFirstKind(0,0.1,p)
        zvec = ChebyshevPointsFirstKind(0,0.1,p)
        W = unscaledWeightsFirstKind(p)
        
        xm,ym,zm = np.meshgrid(xvec,yvec,zvec, indexing='ij')
#         f = xm**2 + ym**3 + zm**5
        f = np.exp(-xm)*np.exp(-2*ym)*np.exp(-5*zm)
#         print(f)
#         print('f[:,0,0]')
#         print(f[:,0,0])
#         f = np.zeros((p,p,p))
#         for i in range(p):
#             for j in range(p):
#                 for k in range(p):
#                     f[i,j,k] = xvec[i]**2 + yvec[j]**3 + zvec[k]**4
#         print(f)          
        w = weights3DFirstKind(0,0.1, p, 0,0.1, p, 0,0.1, p, W)
        
#         xgrad = 2*xm**1 * ym**3 * zm**4
#         ygrad = 3*xm**2 * ym**2 * zm**4
#         zgrad = 4*xm**2 * ym**3 * zm**3

#         xgrad = 2*xm**1
#         ygrad = 3*ym**2
#         zgrad = 5*zm**4
        xgrad = -f
        ygrad = -2*f
        zgrad = -5*f

        DopenX = computeDerivativeMatrix(0,0.1,p)
        DopenY = computeDerivativeMatrix(0,0.1,p)
        DopenZ = computeDerivativeMatrix(0,0.1,p)
#         ccGrad = ChebGradient3D(0,0.1,0,0.1,0,0.1, p, f)
        ccGrad = ChebGradient3D(DopenX,DopenY,DopenZ, p, f)
        
#         print('ccGrad[0]')
#         print(ccGrad[0])
#         print()
#         print('ccGrad[1]')
#         print(ccGrad[1])
#         print()
#         print('ccGrad[2]')
#         print(ccGrad[2])
# #         
#         print('xgrad')
#         print(xgrad)
        
#         print('CC Gradient_y')
#         print(ccGrad[1])
#         print()
#         print("Actualy gradient_y")
#         print(ygrad)

        print('Errors...') 
        print(np.max(abs(xgrad-ccGrad[0])))
        print(np.max(abs(ygrad-ccGrad[1])))
        print(np.max(abs(zgrad-ccGrad[2])))
#         print(zgrad)
#         print(zgrad-ccGrad[2])
#         print()
#         
#         print(ccGrad[0])
    
    def testIntegratingGradient(self):
        p = 4
        xvec = ChebyshevPointsFirstKind(0,0.1,p)
        yvec = ChebyshevPointsFirstKind(0,0.1,p)
        zvec = ChebyshevPointsFirstKind(0,0.1,p)
        W = unscaledWeightsFirstKind(p)
        
        


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()