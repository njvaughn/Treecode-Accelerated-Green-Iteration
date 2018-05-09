'''
Created on Apr 17, 2018

@author: nathanvaughn
'''
import sys
sys.path.append('../src/utilities')
import unittest
import numpy as np

from CuspIntegrationTests import *


class TestCuspIntegration(unittest.TestCase):

    def testCuspFunctionValues(self):
        self.assertEqual(f(2.4), np.exp(-2.4), 'f not evaluating to expected value')
        self.assertEqual(f(-2.4), np.exp(-2.4), 'f not evaluating to expected value')
        self.assertEqual(f(1.625), np.exp(-1.625), 'f not even')
        self.assertEqual(f(41), np.exp(-41), 'f not even')
        
    def testAnalyticIntegration(self):
        self.assertAlmostEqual(analyticIntegral(0,10), np.exp(0)-np.exp(-10),12, 'integral_0^10 not evaluated correctly.')
        self.assertAlmostEqual(analyticIntegral(-10,10), 2*(np.exp(0)-np.exp(-10)),12, 'integral_-10^10 not evaluated correctly.')
        self.assertAlmostEqual(analyticIntegral(-5,1), (2*np.exp(0)-np.exp(-1)-np.exp(-5)),12, 'integral_-5^1 not evaluated correctly.')
        
    def testCuspLocationConstraints(self):
        '''
        Verify that the midpoints or endpoints contain a value very close to 0 when they are
        supposed to.  
        '''
        xlow  = -5
        xhigh = 5
        N = 10
        midpoints,endpoints = generateUniformMesh(xlow,xhigh,N)
        self.assertLess(min(abs(0-endpoints)),1e-15,"0 should be an endpoint when N even and xlow=-xhigh")

        midpoints,endpoints = generateUniformMesh(xlow,xhigh,N+1)    
        self.assertLess(min(abs(0-midpoints)),1e-15,"0 should be a midpoint when N odd and xlow=-xhigh")
            
        



if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testCuspIntegration']
    unittest.main()