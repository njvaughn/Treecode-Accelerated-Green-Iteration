'''
Created on Jul 18, 2018

@author: natha
'''
import unittest
import numpy as np

import time

import matplotlib
matplotlib.use("TKAgg")
import matplotlib.pyplot as plt

from meshUtilities import weights, ChebyshevPoints, interpolator1Dchebyshev, interpolator1Duniform,\
    interpolator2Dchebyshev, interpolator2Dchebyshev_oneStep
# from interpolation import interpolator1D

class TestInterpolation(unittest.TestCase):
    

    @classmethod
    def setUpClass(self):
        
        pass

    def tearDown(self):
        pass
    
    @unittest.skip("Skipping Nested/Recursive equivalence")
    def test2DInterpolationEquivalence(self):
        def func(x,y):
#             return abs(x) # cusp, all should struggle
#             return x**2-y**2  # Polynomial order 2, should capture exactly with 3 Cheb ChebPoints
            return x**7-y**5  # Polynomial order 7, should capture exactly with 8 Cheb ChebPoints
#             return np.exp(-x**2-y**2) # smooth, but not polynomial.  Should improve with more poitns
#             return np.exp(-np.sqrt(x**2+y**2)) # smooth, but not polynomial.  Should improve with more poitns
#             return 1/(x-1.1) # approaching a singularity that is just outside the domain
#         title = r"$f(x) = x^2 - y^2$"
        title = r"$f(x,y) = x^7 - y^5$"
#         title = r"$f(x) = |x|$"
#         title = r"$f(x,y) = exp(-x^2-y^2)$"
#         title = r"$f(x,y) = exp(-\sqrt{x^2+y^2})$"
#         title = r"$f(x) = 1/(x-1.1)$"
        npoints = 100
        testXpoints = np.linspace(-1,1,npoints)
        testYpoints = np.linspace(-1,1,npoints)
        testf = np.zeros((npoints,npoints))
        for i in range(npoints):
            for j in range(npoints):
                testf[i,j] = func(testXpoints[i],testYpoints[j])
                
        nChebPts = 7
        
        xCheb = ChebyshevPoints(-1, 1, nChebPts)
        yCheb = ChebyshevPoints(-1, 1, nChebPts)
        fCheb = np.zeros((nChebPts,nChebPts)) 
        for i in range(nChebPts):
            for j in range(nChebPts):
                fCheb[i,j] = func(xCheb[i],yCheb[j])  
        start = time.time()    
        P_recursive = interpolator2Dchebyshev(xCheb, yCheb, fCheb)
        print('Time to generate P with recursive function:      ', time.time()-start)
        
        start = time.time()
        P_nested = interpolator2Dchebyshev_oneStep(xCheb, yCheb, fCheb)
        print('Time to generate P with nested function:         ', time.time()-start)
    
        
        
        xg, yg = np.meshgrid(testXpoints, testYpoints, indexing='ij')
        start = time.time()
        interpResultA = P_recursive(xg, yg)
        print('Time to evaluate P with recursive function:      ', time.time()-start)
        start = time.time()
        interpResultB = P_nested(xg, yg)
        print('Time to evaluate P with nested function:         ', time.time()-start)
        
        plt.figure()
        img =plt.imshow(np.transpose(testf))
        img.axes.get_xaxis().set_visible(False)
        img.axes.get_yaxis().set_visible(False)
        plt.colorbar(img,fraction=0.046, pad=0.04)
        plt.title(title)
        
        
        f1, (ax1, ax2, ax3) = plt.subplots(1, 3,figsize=(16,6))
        
        img1 = ax1.imshow(np.transpose( np.log10(abs(interpResultA-testf)) ))
#         img1.axis('off')
        img1.axes.get_xaxis().set_visible(False)
        img1.axes.get_yaxis().set_visible(False)
        plt.colorbar(img1,ax=ax1,fraction=0.046, pad=0.04)
        ax1.set_title('Using Recursive function')
        
        img2 = ax2.imshow(np.transpose(np.log10(abs(interpResultB-testf))))
#         img2.axis('off')
        img2.axes.get_xaxis().set_visible(False)
        img2.axes.get_yaxis().set_visible(False)
        plt.colorbar(img2,ax=ax2,fraction=0.046, pad=0.04)
        ax2.set_title('Using Nested Function')
        
        img3 = ax3.imshow(np.transpose(abs(interpResultB-interpResultA)))
#         img2.axis('off')
        img3.axes.get_xaxis().set_visible(False)
        img3.axes.get_yaxis().set_visible(False)
        plt.colorbar(img3,ax=ax3,fraction=0.046, pad=0.04)
        ax3.set_title('Difference')
 
        
        plt.suptitle('Chebyshev interpolation errors for '+ title)
        plt.tight_layout()
        plt.show()
    
    
#     @unittest.skip("Skipping 2D Convergence")
    def test2DInterpolationConvergence(self):
        def func(x,y):
#             return abs(x) # cusp, all should struggle
#             return x**2-y**2  # Polynomial order 2, should capture exactly with 3 Cheb ChebPoints
#             return x**7-y**5  # Polynomial order 7, should capture exactly with 8 Cheb ChebPoints
#             return np.exp(-x**2-y**2) # smooth, but not polynomial.  Should improve with more poitns
            return np.exp(-np.sqrt(x**2+y**2)) # smooth, but not polynomial.  Should improve with more poitns
#             return 1/(x-1.1) # approaching a singularity that is just outside the domain
#         title = r"$f(x) = x^2 - y^2$"
#         title = r"$f(x,y) = x^7 - y^5$"
#         title = r"$f(x) = |x|$"
#         title = r"$f(x,y) = exp(-x^2-y^2)$"
        title = r"$f(x,y) = exp(-\sqrt{x^2+y^2})$"
#         title = r"$f(x) = 1/(x-1.1)$"
        npoints = 200
        xmin = -1
        ymin = -1
        xmax =  0
        ymax =  0
        testXpoints = np.linspace(xmin,xmax,npoints)
        testYpoints = np.linspace(ymin,ymax,npoints)
        testf = np.zeros((npoints,npoints))
        for i in range(npoints):
            for j in range(npoints):
                testf[i,j] = func(testXpoints[i],testYpoints[j])
                
        nChebPtsA = 4
        nChebPtsB = 8
        nChebPtsC = 12
        
        xCheb = ChebyshevPoints(xmin, xmax, nChebPtsA)
        yCheb = ChebyshevPoints(xmin, xmax, nChebPtsA)
        fCheb = np.zeros((nChebPtsA,nChebPtsA)) 
        for i in range(nChebPtsA):
            for j in range(nChebPtsA):
                fCheb[i,j] = func(xCheb[i],yCheb[j])      
        PA = interpolator2Dchebyshev_oneStep(xCheb, yCheb, fCheb)
        
        
        xCheb = ChebyshevPoints(xmin, xmax, nChebPtsB)
        yCheb = ChebyshevPoints(xmin, xmax, nChebPtsB)
        fCheb = np.zeros((nChebPtsB,nChebPtsB)) 
        for i in range(nChebPtsB):
            for j in range(nChebPtsB):
                fCheb[i,j] = func(xCheb[i],yCheb[j])      
        PB = interpolator2Dchebyshev_oneStep(xCheb, yCheb, fCheb)
        
        xCheb = ChebyshevPoints(xmin, xmax, nChebPtsC)
        yCheb = ChebyshevPoints(xmin, xmax, nChebPtsC)
        fCheb = np.zeros((nChebPtsC,nChebPtsC)) 
        for i in range(nChebPtsC):
            for j in range(nChebPtsC):
                fCheb[i,j] = func(xCheb[i],yCheb[j])      
        PC = interpolator2Dchebyshev_oneStep(xCheb, yCheb, fCheb)
        
        
        xg, yg = np.meshgrid(testXpoints, testYpoints, indexing='ij')
        interpResultA = PA(xg, yg)
        interpResultB = PB(xg, yg)
        interpResultC = PC(xg, yg)
#         print(testf)
#         print()
        
        
#         plt.figure()
#         img =plt.imshow(np.transpose(testf))
#         img.axes.get_xaxis().set_visible(False)
#         img.axes.get_yaxis().set_visible(False)
#         plt.colorbar(img,fraction=0.046, pad=0.04)
#         plt.title(title)
        
        
        f1, (ax1, ax2, ax3) = plt.subplots(1, 3,figsize=(16,6))
        
        img1 = ax1.imshow(np.transpose( np.log10(abs(interpResultA-testf)) ), cmap="inferno")
#         img1.axis('off')
        img1.axes.get_xaxis().set_visible(False)
        img1.axes.get_yaxis().set_visible(False)
        plt.colorbar(img1,ax=ax1,fraction=0.046, pad=0.04)
        ax1.set_title('Using %i Chebyshev Points'%nChebPtsA)
        
        img2 = ax2.imshow(np.transpose(np.log10(abs(interpResultB-testf))), cmap="inferno")
#         img2.axis('off')
        img2.axes.get_xaxis().set_visible(False)
        img2.axes.get_yaxis().set_visible(False)
        plt.colorbar(img2,ax=ax2,fraction=0.046, pad=0.04)
        ax2.set_title('Using %i Chebyshev Points'%nChebPtsB)
        
        img3 = ax3.imshow(np.transpose(np.log10(abs(interpResultC-testf))), cmap="inferno")
#         img3.axis('off')
        img3.axes.get_xaxis().set_visible(False)
        img3.axes.get_yaxis().set_visible(False)
        plt.colorbar(img3,ax=ax3,fraction=0.046, pad=0.04)
        ax3.set_title('Using %i Chebyshev Points'%nChebPtsC)
        
        plt.suptitle('Chebyshev interpolation errors for '+ title)
        plt.tight_layout()
        
        
        f2, (ax1, ax2, ax3) = plt.subplots(1, 3,figsize=(16,6))
        
        img1 = ax1.imshow(np.transpose( interpResultA ))
#         img1.axis('off')
        img1.axes.get_xaxis().set_visible(False)
        img1.axes.get_yaxis().set_visible(False)
        plt.colorbar(img1,ax=ax1,fraction=0.046, pad=0.04)
        ax1.set_title('Using %i Chebyshev Points'%nChebPtsA)
        
        img2 = ax2.imshow(np.transpose(interpResultB))
#         img2.axis('off')
        img2.axes.get_xaxis().set_visible(False)
        img2.axes.get_yaxis().set_visible(False)
        plt.colorbar(img2,ax=ax2,fraction=0.046, pad=0.04)
        ax2.set_title('Using %i Chebyshev Points'%nChebPtsB)
        
        img3 = ax3.imshow(np.transpose(interpResultC))
#         img3.axis('off')
        img3.axes.get_xaxis().set_visible(False)
        img3.axes.get_yaxis().set_visible(False)
        plt.colorbar(img3,ax=ax3,fraction=0.046, pad=0.04)
        ax3.set_title('Using %i Chebyshev Points'%nChebPtsC)
        
        plt.suptitle('Interpolants for '+ title)
        plt.tight_layout()
        

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()