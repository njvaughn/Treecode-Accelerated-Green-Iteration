'''
Created on Jul 18, 2018

@author: natha
'''
import unittest
import numpy as np
import sys
sys.path.append('../src/dataStructures')
sys.path.append('../src/utilities')
import time

import matplotlib
matplotlib.use("TKAgg")
import matplotlib.pyplot as plt

from meshUtilities import weights, ChebyshevPoints, interpolator1Dchebyshev, interpolator1Duniform,\
    interpolator2Dchebyshev, interpolator2Dchebyshev_oneStep, interpolator3Dchebyshev
# from interpolation import interpolator1D

class TestInterpolation(unittest.TestCase):
    

    @classmethod
    
    
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
    @unittest.skip("Skipping the convergence plotter")
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
#         xmin = -0.5
#         ymin = -0.5
#         xmax =  0.5
#         ymax =  0.5
        xmin = -1
        ymin = 0
        xmax =  0
        ymax =  1
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
        yCheb = ChebyshevPoints(ymin, ymax, nChebPtsA)
        xChebMeshA, yChebMeshA = np.meshgrid(xCheb,yCheb)

        fCheb = np.zeros((nChebPtsA,nChebPtsA)) 
        for i in range(nChebPtsA):
            for j in range(nChebPtsA):
                fCheb[i,j] = func(xCheb[i],yCheb[j])      
        PA = interpolator2Dchebyshev_oneStep(xCheb, yCheb, fCheb)
        
        
        xCheb = ChebyshevPoints(xmin, xmax, nChebPtsB)
        yCheb = ChebyshevPoints(ymin, ymax, nChebPtsB)
        xChebMeshB, yChebMeshB = np.meshgrid(xCheb,yCheb)
        fCheb = np.zeros((nChebPtsB,nChebPtsB)) 
        for i in range(nChebPtsB):
            for j in range(nChebPtsB):
                fCheb[i,j] = func(xCheb[i],yCheb[j])      
        PB = interpolator2Dchebyshev_oneStep(xCheb, yCheb, fCheb)
        
        xCheb = ChebyshevPoints(xmin, xmax, nChebPtsC)
        yCheb = ChebyshevPoints(ymin, ymax, nChebPtsC)
        xChebMeshC, yChebMeshC = np.meshgrid(xCheb,yCheb)
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
        
#         print(xChebMeshA)
#         print()
#         print(yChebMeshA)
#         print()
#         print(np.log10(abs(interpResultA-testf)))
        
        f2, (ax11, ax22, ax33) = plt.subplots(1, 3,figsize=(16,5))
#         ax11.scatter(xg.flatten(),yg.flatten(),color=np.log10(abs(interpResultA-testf)).flatten())
        left = ax11.scatter(xg.flatten(),yg.flatten(),marker='.',
                     c=np.log10(abs(interpResultA-testf)).flatten(),cmap="inferno",vmin=-11, vmax=-1)
        ax11.scatter(xChebMeshA.flatten(),yChebMeshA.flatten(),c='k')
        ax11.set_title('Using %i Chebyshev Points'%nChebPtsA)
        middle = ax22.scatter(xg.flatten(),yg.flatten(),marker='.',
                     c=np.log10(abs(interpResultB-testf)).flatten(),cmap="inferno",vmin=-11, vmax=-1)
        ax22.scatter(xChebMeshB.flatten(),yChebMeshB.flatten(),c='k')
        ax22.set_title('Using %i Chebyshev Points'%nChebPtsB)
        right = ax33.scatter(xg.flatten(),yg.flatten(),marker='.',
                     c=np.log10(abs(interpResultC-testf)).flatten(),cmap="inferno",vmin=-11, vmax=-1)
        ax33.scatter(xChebMeshC.flatten(),yChebMeshC.flatten(),c='k')
        ax33.set_title('Using %i Chebyshev Points'%nChebPtsC)
        
        plt.colorbar(left,ax=ax11,fraction=0.046, pad=0.04)
        plt.colorbar(middle,ax=ax22,fraction=0.046, pad=0.04)
        plt.colorbar(right,ax=ax33,fraction=0.046, pad=0.04)
        
        
        f2.suptitle('Chebyshev Meshes')
        f2.tight_layout()
        f2.subplots_adjust(top=0.8)
        
        
#         f1, (ax1, ax2, ax3) = plt.subplots(1, 3,figsize=(16,6))
#         img1 = ax1.imshow(np.transpose( np.log10(abs(interpResultA-testf)) ),vmin=-11, vmax=-1, cmap="inferno")
# #         img1.axis('off')
#         img1.axes.get_xaxis().set_visible(False)
#         img1.axes.get_yaxis().set_visible(False)
        
# #         plt.clim(-11,-2,ax=ax1)
#         ax1.set_title('Using %i Chebyshev Points'%nChebPtsA)
#         
#         img2 = ax2.imshow(np.transpose(np.log10(abs(interpResultB-testf))),vmin=-11, vmax=-1, cmap="inferno")
# #         img2.axis('off')
#         img2.axes.get_xaxis().set_visible(False)
#         img2.axes.get_yaxis().set_visible(False)
#         plt.colorbar(img2,ax=ax2,fraction=0.046, pad=0.04)
#         ax2.set_title('Using %i Chebyshev Points'%nChebPtsB)
#         
#         img3 = ax3.imshow(np.transpose(np.log10(abs(interpResultC-testf))),vmin=-11, vmax=-1, cmap="inferno")
# #         img3.axis('off')
#         img3.axes.get_xaxis().set_visible(False)
#         img3.axes.get_yaxis().set_visible(False)
#         plt.colorbar(img3,ax=ax3,fraction=0.046, pad=0.04)
#         ax3.set_title('Using %i Chebyshev Points'%nChebPtsC)
#         
#         f1.suptitle('Chebyshev interpolation errors for '+ title)
#         f1.tight_layout()
        plt.show()
        
#         
#         f2, (ax1, ax2, ax3) = plt.subplots(1, 3,figsize=(16,6))
#         
#         img1 = ax1.imshow(np.transpose( interpResultA ))
# #         img1.axis('off')
#         img1.axes.get_xaxis().set_visible(False)
#         img1.axes.get_yaxis().set_visible(False)
#         plt.colorbar(img1,ax=ax1,fraction=0.046, pad=0.04)
#         ax1.set_title('Using %i Chebyshev Points'%nChebPtsA)
#         
#         img2 = ax2.imshow(np.transpose(interpResultB))
# #         img2.axis('off')
#         img2.axes.get_xaxis().set_visible(False)
#         img2.axes.get_yaxis().set_visible(False)
#         plt.colorbar(img2,ax=ax2,fraction=0.046, pad=0.04)
# #         plt.clim(-11, -2)
#         ax2.set_title('Using %i Chebyshev Points'%nChebPtsB)
#         
#         img3 = ax3.imshow(np.transpose(interpResultC))
# #         img3.axis('off')
#         img3.axes.get_xaxis().set_visible(False)
#         img3.axes.get_yaxis().set_visible(False)
#         plt.colorbar(img3,ax=ax3,fraction=0.046, pad=0.04)
# #         plt.clim(-11, -2)
#         ax3.set_title('Using %i Chebyshev Points'%nChebPtsC)
#         
#         plt.suptitle('Interpolants for '+ title)
#         plt.tight_layout()
#         plt.show()





    @unittest.skip("Skipping the convergence plotter")
    def testOrderVersusRefinement(self):
        def func(x,y):
            r = np.sqrt(x**2+y**2)
#             return np.exp(-np.sqrt(x**2+y**2))
#             return np.exp(-r)/(r) 
            return np.exp(-r) 

#         def func(x,y,xt,yt):
#             r = np.sqrt((x-xt)**2+(y-yt)**2)
#             return np.exp(-r)/(r) 

#         title = r"$f(x,y) = exp(-\sqrt{x^2+y^2})$"
#         title = r"$f(r) = exp(-r)/r$"
        title = r"$f(r) = exp(-r)$"

        npoints = 201

        xmin = -1
        ymin = 0
        xmax =  0
        ymax =  1
        testXpoints = np.linspace(xmin,xmax,npoints)
        testYpoints = np.linspace(ymin,ymax,npoints)
        testf = np.zeros((npoints,npoints))
        for i in range(npoints):
            for j in range(npoints):
                testf[i,j] = func(testXpoints[i],testYpoints[j])
                
        nChebPtsA = 4
        nChebPtsB = 8
        nChebPtsC = 16
        
        
        
        xCheb = ChebyshevPoints(xmin, xmax, nChebPtsA)
        yCheb = ChebyshevPoints(ymin, ymax, nChebPtsA)
        xChebMesh1A, yChebMesh1A = np.meshgrid(xCheb,yCheb)

        fCheb = np.zeros((nChebPtsA,nChebPtsA)) 
        for i in range(nChebPtsA):
            for j in range(nChebPtsA):
                fCheb[i,j] = func(xCheb[i],yCheb[j])      
        P1A = interpolator2Dchebyshev_oneStep(xCheb, yCheb, fCheb)
        
        
        xCheb = ChebyshevPoints(xmin, xmax, nChebPtsB)
        yCheb = ChebyshevPoints(ymin, ymax, nChebPtsB)
        xChebMesh1B, yChebMesh1B = np.meshgrid(xCheb,yCheb)
        fCheb = np.zeros((nChebPtsB,nChebPtsB)) 
        for i in range(nChebPtsB):
            for j in range(nChebPtsB):
                fCheb[i,j] = func(xCheb[i],yCheb[j])      
        P1B = interpolator2Dchebyshev_oneStep(xCheb, yCheb, fCheb)
        
        xCheb = ChebyshevPoints(xmin, xmax, nChebPtsC)
        yCheb = ChebyshevPoints(ymin, ymax, nChebPtsC)
        xChebMesh1C, yChebMesh1C = np.meshgrid(xCheb,yCheb)
        fCheb = np.zeros((nChebPtsC,nChebPtsC)) 
        for i in range(nChebPtsC):
            for j in range(nChebPtsC):
                fCheb[i,j] = func(xCheb[i],yCheb[j])      
        P1C = interpolator2Dchebyshev_oneStep(xCheb, yCheb, fCheb)
        
        
        xg1, yg1 = np.meshgrid(testXpoints, testYpoints, indexing='ij')
        interpResult1A = P1A(xg1, yg1)
        interpResult1B = P1B(xg1, yg1)
        interpResult1C = P1C(xg1, yg1)
        
        
        
        
        f2, ((ax11, ax12, ax13),(ax21, ax22, ax23)) = plt.subplots(2, 3,figsize=(12,6))

        left = ax11.scatter(xg1.flatten(),yg1.flatten(),marker='.',
                     c=np.log10(abs(interpResult1A-testf)).flatten(),cmap="inferno",vmin=-11, vmax=-1)
        ax11.scatter(xChebMesh1A.flatten(),yChebMesh1A.flatten(),c='k',s=1)
        ax11.set_title('Using %i Chebyshev Points'%nChebPtsA)
        middle = ax12.scatter(xg1.flatten(),yg1.flatten(),marker='.',
                     c=np.log10(abs(interpResult1B-testf)).flatten(),cmap="inferno",vmin=-11, vmax=-1)
        ax12.scatter(xChebMesh1B.flatten(),yChebMesh1B.flatten(),c='k',s=1)
        ax12.set_title('Using %i Chebyshev Points'%nChebPtsB)
        right = ax13.scatter(xg1.flatten(),yg1.flatten(),marker='.',
                     c=np.log10(abs(interpResult1C-testf)).flatten(),cmap="inferno",vmin=-11, vmax=-1)
        ax13.scatter(xChebMesh1C.flatten(),yChebMesh1C.flatten(),c='k',s=1)
        ax13.set_title('Using %i Chebyshev Points'%nChebPtsC)
        
#         plt.colorbar(left,ax=ax11,fraction=0.046, pad=0.04)
#         plt.colorbar(middle,ax=ax12,fraction=0.046, pad=0.04)
        plt.colorbar(right,ax=ax13,fraction=0.046, pad=0.04)
        
        
        """ Test refinement using a fixed number of chebyshev points per cell """
#         nChebPtsR = 4
        
        counter=1
        for nChebPts in [int(nChebPtsA/2),int(nChebPtsB/2),int(nChebPtsC/2)]:
            if counter == 1:
                currentAxes = ax21
            elif counter == 2:
                currentAxes = ax22
            elif counter == 3:
                currentAxes = ax23
        
            xminA = -1
            xmaxA = -0.5
            yminA =  0
            ymaxA =  0.5
            testXpointsA = np.linspace(xminA,xmaxA,int(npoints/2))
            testYpointsA = np.linspace(yminA,ymaxA,int(npoints/2))
            testfA = np.zeros((int(npoints/2),int(npoints/2)))
            for i in range(int(npoints/2)):
                for j in range(int(npoints/2)):
                    testfA[i,j] = func(testXpointsA[i],testYpointsA[j])
                    
            xChebA = ChebyshevPoints(xminA, xmaxA, nChebPts)
            yChebA = ChebyshevPoints(yminA, ymaxA, nChebPts)
            xChebMeshA, yChebMeshA = np.meshgrid(xChebA,yChebA)
    
            fChebA = np.zeros((nChebPts,nChebPts)) 
            for i in range(nChebPts):
                for j in range(nChebPts):
                    fChebA[i,j] = func(xChebA[i],yChebA[j])      
            PA = interpolator2Dchebyshev_oneStep(xChebA, yChebA, fChebA)
            xg, yg = np.meshgrid(testXpointsA, testYpointsA, indexing='ij')
            interpResultA = PA(xg, yg)
            currentAxes.scatter(xg.flatten(),yg.flatten(),marker='.',
                     c=np.log10(abs(interpResultA-testfA)).flatten(),cmap="inferno",vmin=-11, vmax=-1,s=1)
            currentAxes.scatter(xChebMeshA.flatten(),yChebMeshA.flatten(),c='k',s=3)
            
            
            xminB = -1
            xmaxB =  -0.5
            yminB =  0.5
            ymaxB =  1
            testXpointsB = np.linspace(xminB,xmaxB,int(npoints/2))
            testYpointsB = np.linspace(yminB,ymaxB,int(npoints/2))
            testfB = np.zeros((int(npoints/2),int(npoints/2)))
            for i in range(int(npoints/2)):
                for j in range(int(npoints/2)):
                    testfB[i,j] = func(testXpointsB[i],testYpointsB[j])
            
            xChebB = ChebyshevPoints(xminB, xmaxB, nChebPts)
            yChebB = ChebyshevPoints(yminB, ymaxB, nChebPts)
            xChebMeshB, yChebMeshB = np.meshgrid(xChebB,yChebB)
    
            fChebB = np.zeros((nChebPts,nChebPts)) 
            for i in range(nChebPts):
                for j in range(nChebPts):
                    fChebB[i,j] = func(xChebB[i],yChebB[j])      
            PB = interpolator2Dchebyshev_oneStep(xChebB, yChebB, fChebB)
            xg, yg = np.meshgrid(testXpointsB, testYpointsB, indexing='ij')
            interpResultB = PB(xg, yg)
            currentAxes.scatter(xg.flatten(),yg.flatten(),marker='.',
                     c=np.log10(abs(interpResultB-testfB)).flatten(),cmap="inferno",vmin=-11, vmax=-1,s=1)
            currentAxes.scatter(xChebMeshB.flatten(),yChebMeshB.flatten(),c='k',s=3)

            
            
            xminC = -0.5
            xmaxC =  0
            yminC =  0
            ymaxC =  0.5
            testXpointsC = np.linspace(xminC,xmaxC,npoints/2)
            testYpointsC = np.linspace(yminC,ymaxC,npoints/2)
            testfC = np.zeros((int(npoints/2),int(npoints/2)))
            for i in range(int(npoints/2)):
                for j in range(int(npoints/2)):
                    testfC[i,j] = func(testXpointsC[i],testYpointsC[j])
            
            xChebC = ChebyshevPoints(xminC, xmaxC, nChebPts)
            yChebC = ChebyshevPoints(yminC, ymaxC, nChebPts)
            xChebMeshC, yChebMeshC = np.meshgrid(xChebC,yChebC)
    
            fChebC = np.zeros((nChebPts,nChebPts)) 
            for i in range(nChebPts):
                for j in range(nChebPts):
                    fChebC[i,j] = func(xChebC[i],yChebC[j])      
            PC = interpolator2Dchebyshev_oneStep(xChebC, yChebC, fChebC)
            xg, yg = np.meshgrid(testXpointsC, testYpointsC, indexing='ij')
            interpResultC = PC(xg, yg)
            currentAxes.scatter(xg.flatten(),yg.flatten(),marker='.',
                     c=np.log10(abs(interpResultC-testfC)).flatten(),cmap="inferno",vmin=-11, vmax=-1,s=1)
            currentAxes.scatter(xChebMeshC.flatten(),yChebMeshC.flatten(),c='k',s=3)

            
            
            xminD = -0.5
            xmaxD =  0
            yminD =  0.5
            ymaxD =  1
            testXpointsD = np.linspace(xminD,xmaxD,npoints/2)
            testYpointsD = np.linspace(yminD,ymaxD,npoints/2)
            testfD = np.zeros((int(npoints/2),int(npoints/2)))
            for i in range(int(npoints/2)):
                for j in range(int(npoints/2)):
                    testfD[i,j] = func(testXpointsD[i],testYpointsD[j])
                    
            xChebD = ChebyshevPoints(xminD, xmaxD, nChebPts)
            yChebD = ChebyshevPoints(yminD, ymaxD, nChebPts)
            xChebMeshD, yChebMeshD = np.meshgrid(xChebD,yChebD)
    
            fChebD = np.zeros((nChebPts,nChebPts)) 
            for i in range(nChebPts):
                for j in range(nChebPts):
                    fChebD[i,j] = func(xChebD[i],yChebD[j])      
            PD = interpolator2Dchebyshev_oneStep(xChebD, yChebD, fChebD)
            
            xg, yg = np.meshgrid(testXpointsD, testYpointsD, indexing='ij')
            interpResultD = PD(xg, yg)
            currentAxes.scatter(xg.flatten(),yg.flatten(),marker='.',
                     c=np.log10(abs(interpResultD-testfD)).flatten(),cmap="inferno",vmin=-11, vmax=-1,s=1)
            currentAxes.scatter(xChebMeshD.flatten(),yChebMeshD.flatten(),c='k',s=3)
            
                
            counter += 1
            
            
            
        plt.colorbar(right,ax=ax23,fraction=0.046, pad=0.04)
        
        
#         f2.suptitle('Chebyshev Meshes')
        f2.suptitle(title)
#         f2.tight_layout()
        ax11.set_aspect(1.0)
        ax12.set_aspect(1.0)
        ax13.set_aspect(1.0)
        ax21.set_aspect(1.0)
        ax22.set_aspect(1.0)
        ax23.set_aspect(1.0)
        f2.tight_layout()
        f2.subplots_adjust(top=0.8)
        
        

        plt.show()
     
    
    
    @unittest.skip("Skipping singularity at target-point test")   
    def testSingularityAtTargetPoint(self):
#         def func(x,y):
#             r = np.sqrt(x**2+y**2)
# #             return np.exp(-np.sqrt(x**2+y**2))
#             return np.exp(-r)/(r) 
# #             return np.exp(-r) 

        def func(x,y,xt,yt):
            r = np.sqrt((x-xt)**2+(y-yt)**2)
            if r>0.0:
                return np.exp(-r)#/(r)
            else:
                return 0.0

#         title = r"$f(x,y) = exp(-\sqrt{x^2+y^2})$"
#         title = r"$f(r) = exp(-r)/r$"
        title = r"$f(r) = exp(-r)$"

        npoints = 201

        xmin = -1
        ymin = 0
        xmax =  0
        ymax =  1
        testXpoints = np.linspace(xmin,xmax,npoints)
        testYpoints = np.linspace(ymin,ymax,npoints)
        testf = np.zeros((npoints,npoints))
        
                
        nChebPtsA = 4
        nChebPtsB = 8
        nChebPtsC = 16
        
        
        
        xCheb = ChebyshevPoints(xmin, xmax, nChebPtsA)
        yCheb = ChebyshevPoints(ymin, ymax, nChebPtsA)
#         xt = xCheb[3]
#         yt = yCheb[3]
        
        # center cusp at (0,0), divide at (-1/2, 1/2)
        xt = 0
        yt = 0
        xd = -1/2
        yd = 1/2
        xChebMesh1A, yChebMesh1A = np.meshgrid(xCheb,yCheb)
        
        for i in range(npoints):
            for j in range(npoints):
                testf[i,j] = func(testXpoints[i],testYpoints[j],xt,yt)

        fCheb = np.zeros((nChebPtsA,nChebPtsA)) 
        for i in range(nChebPtsA):
            for j in range(nChebPtsA):
                fCheb[i,j] = func(xCheb[i],yCheb[j],xt,yt)      
        P1A = interpolator2Dchebyshev_oneStep(xCheb, yCheb, fCheb)
        
        
        xCheb = ChebyshevPoints(xmin, xmax, nChebPtsB)
        yCheb = ChebyshevPoints(ymin, ymax, nChebPtsB)
        xChebMesh1B, yChebMesh1B = np.meshgrid(xCheb,yCheb)
        fCheb = np.zeros((nChebPtsB,nChebPtsB)) 
        for i in range(nChebPtsB):
            for j in range(nChebPtsB):
                fCheb[i,j] = func(xCheb[i],yCheb[j],xt,yt)      
        P1B = interpolator2Dchebyshev_oneStep(xCheb, yCheb, fCheb)
        
        xCheb = ChebyshevPoints(xmin, xmax, nChebPtsC)
        yCheb = ChebyshevPoints(ymin, ymax, nChebPtsC)
        xChebMesh1C, yChebMesh1C = np.meshgrid(xCheb,yCheb)
        fCheb = np.zeros((nChebPtsC,nChebPtsC)) 
        for i in range(nChebPtsC):
            for j in range(nChebPtsC):
                fCheb[i,j] = func(xCheb[i],yCheb[j],xt,yt)      
        P1C = interpolator2Dchebyshev_oneStep(xCheb, yCheb, fCheb)
        
        
        xg1, yg1 = np.meshgrid(testXpoints, testYpoints, indexing='ij')
        interpResult1A = P1A(xg1, yg1)
        interpResult1B = P1B(xg1, yg1)
        interpResult1C = P1C(xg1, yg1)
        
        
        
        
        f2, ((ax11, ax12, ax13),(ax21, ax22, ax23)) = plt.subplots(2, 3,figsize=(12,6))

        left = ax11.scatter(xg1.flatten(),yg1.flatten(),marker='.',
                     c=np.log10(abs(interpResult1A-testf)).flatten(),cmap="inferno",vmin=-11, vmax=0)
        ax11.scatter(xChebMesh1A.flatten(),yChebMesh1A.flatten(),c='k',s=1)
        ax11.set_title('Using %i Chebyshev Points'%nChebPtsA)
        middle = ax12.scatter(xg1.flatten(),yg1.flatten(),marker='.',
                     c=np.log10(abs(interpResult1B-testf)).flatten(),cmap="inferno",vmin=-11, vmax=0)
        ax12.scatter(xChebMesh1B.flatten(),yChebMesh1B.flatten(),c='k',s=1)
        ax12.set_title('Using %i Chebyshev Points'%nChebPtsB)
        right = ax13.scatter(xg1.flatten(),yg1.flatten(),marker='.',
                     c=np.log10(abs(interpResult1C-testf)).flatten(),cmap="inferno",vmin=-11, vmax=0)
        ax13.scatter(xChebMesh1C.flatten(),yChebMesh1C.flatten(),c='k',s=1)
        ax13.set_title('Using %i Chebyshev Points'%nChebPtsC)
        
#         plt.colorbar(left,ax=ax11,fraction=0.046, pad=0.04)
#         plt.colorbar(middle,ax=ax12,fraction=0.046, pad=0.04)
        plt.colorbar(right,ax=ax13,fraction=0.046, pad=0.04)
        
        
        """ Test refinement using a fixed number of chebyshev points per cell """
#         nChebPtsR = 4
        
        counter=1
        for nChebPts in [int(nChebPtsA/2),int(nChebPtsB/2),int(nChebPtsC/2)]:
            if counter == 1:
                currentAxes = ax21
            elif counter == 2:
                currentAxes = ax22
            elif counter == 3:
                currentAxes = ax23
        
            xminA = -1
            xmaxA = xd
            yminA =  0
            ymaxA =  yd
            testXpointsA = np.linspace(xminA,xmaxA,int(npoints/2))
            testYpointsA = np.linspace(yminA,ymaxA,int(npoints/2))
            testfA = np.zeros((int(npoints/2),int(npoints/2)))
            for i in range(int(npoints/2)):
                for j in range(int(npoints/2)):
                    testfA[i,j] = func(testXpointsA[i],testYpointsA[j],xt,yt)
                    
            xChebA = ChebyshevPoints(xminA, xmaxA, nChebPts)
            yChebA = ChebyshevPoints(yminA, ymaxA, nChebPts)
            xChebMeshA, yChebMeshA = np.meshgrid(xChebA,yChebA)
    
            fChebA = np.zeros((nChebPts,nChebPts)) 
            for i in range(nChebPts):
                for j in range(nChebPts):
                    fChebA[i,j] = func(xChebA[i],yChebA[j],xt,yt)      
            PA = interpolator2Dchebyshev_oneStep(xChebA, yChebA, fChebA)
            xg, yg = np.meshgrid(testXpointsA, testYpointsA, indexing='ij')
            interpResultA = PA(xg, yg)
            currentAxes.scatter(xg.flatten(),yg.flatten(),marker='.',
                     c=np.log10(abs(interpResultA-testfA)).flatten(),cmap="inferno",vmin=-11, vmax=0,s=1)
            currentAxes.scatter(xChebMeshA.flatten(),yChebMeshA.flatten(),c='k',s=3)
            
            
            xminB = -1
            xmaxB =  xd
            yminB =  yd
            ymaxB =  1
            testXpointsB = np.linspace(xminB,xmaxB,int(npoints/2))
            testYpointsB = np.linspace(yminB,ymaxB,int(npoints/2))
            testfB = np.zeros((int(npoints/2),int(npoints/2)))
            for i in range(int(npoints/2)):
                for j in range(int(npoints/2)):
                    testfB[i,j] = func(testXpointsB[i],testYpointsB[j],xt,yt)
            
            xChebB = ChebyshevPoints(xminB, xmaxB, nChebPts)
            yChebB = ChebyshevPoints(yminB, ymaxB, nChebPts)
            xChebMeshB, yChebMeshB = np.meshgrid(xChebB,yChebB)
    
            fChebB = np.zeros((nChebPts,nChebPts)) 
            for i in range(nChebPts):
                for j in range(nChebPts):
                    fChebB[i,j] = func(xChebB[i],yChebB[j],xt,yt)      
            PB = interpolator2Dchebyshev_oneStep(xChebB, yChebB, fChebB)
            xg, yg = np.meshgrid(testXpointsB, testYpointsB, indexing='ij')
            interpResultB = PB(xg, yg)
            currentAxes.scatter(xg.flatten(),yg.flatten(),marker='.',
                     c=np.log10(abs(interpResultB-testfB)).flatten(),cmap="inferno",vmin=-11, vmax=0,s=1)
            currentAxes.scatter(xChebMeshB.flatten(),yChebMeshB.flatten(),c='k',s=3)

            
            
            xminC = xd
            xmaxC =  0
            yminC =  0
            ymaxC =  yd
            testXpointsC = np.linspace(xminC,xmaxC,npoints/2)
            testYpointsC = np.linspace(yminC,ymaxC,npoints/2)
            testfC = np.zeros((int(npoints/2),int(npoints/2)))
            for i in range(int(npoints/2)):
                for j in range(int(npoints/2)):
                    testfC[i,j] = func(testXpointsC[i],testYpointsC[j],xt,yt)
            
            xChebC = ChebyshevPoints(xminC, xmaxC, nChebPts)
            yChebC = ChebyshevPoints(yminC, ymaxC, nChebPts)
            xChebMeshC, yChebMeshC = np.meshgrid(xChebC,yChebC)
    
            fChebC = np.zeros((nChebPts,nChebPts)) 
            for i in range(nChebPts):
                for j in range(nChebPts):
                    fChebC[i,j] = func(xChebC[i],yChebC[j],xt,yt)      
            PC = interpolator2Dchebyshev_oneStep(xChebC, yChebC, fChebC)
            xg, yg = np.meshgrid(testXpointsC, testYpointsC, indexing='ij')
            interpResultC = PC(xg, yg)
            currentAxes.scatter(xg.flatten(),yg.flatten(),marker='.',
                     c=np.log10(abs(interpResultC-testfC)).flatten(),cmap="inferno",vmin=-11, vmax=0,s=1)
            currentAxes.scatter(xChebMeshC.flatten(),yChebMeshC.flatten(),c='k',s=3)

            
            
            xminD =  xd
            xmaxD =  0
            yminD =  yd
            ymaxD =  1
            testXpointsD = np.linspace(xminD,xmaxD,npoints/2)
            testYpointsD = np.linspace(yminD,ymaxD,npoints/2)
            testfD = np.zeros((int(npoints/2),int(npoints/2)))
            for i in range(int(npoints/2)):
                for j in range(int(npoints/2)):
                    testfD[i,j] = func(testXpointsD[i],testYpointsD[j],xt,yt)
                    
            xChebD = ChebyshevPoints(xminD, xmaxD, nChebPts)
            yChebD = ChebyshevPoints(yminD, ymaxD, nChebPts)
            xChebMeshD, yChebMeshD = np.meshgrid(xChebD,yChebD)
    
            fChebD = np.zeros((nChebPts,nChebPts)) 
            for i in range(nChebPts):
                for j in range(nChebPts):
                    fChebD[i,j] = func(xChebD[i],yChebD[j],xt,yt)      
            PD = interpolator2Dchebyshev_oneStep(xChebD, yChebD, fChebD)
            
            xg, yg = np.meshgrid(testXpointsD, testYpointsD, indexing='ij')
            interpResultD = PD(xg, yg)
            currentAxes.scatter(xg.flatten(),yg.flatten(),marker='.',
                     c=np.log10(abs(interpResultD-testfD)).flatten(),cmap="inferno",vmin=-11, vmax=0,s=1)
            currentAxes.scatter(xChebMeshD.flatten(),yChebMeshD.flatten(),c='k',s=3)
            
                
            counter += 1
            
            
            
        plt.colorbar(right,ax=ax23,fraction=0.046, pad=0.04)
        
        
#         f2.suptitle('Chebyshev Meshes')
        f2.suptitle(title)
#         f2.tight_layout()
        ax11.set_aspect(1.0)
        ax12.set_aspect(1.0)
        ax13.set_aspect(1.0)
        ax21.set_aspect(1.0)
        ax22.set_aspect(1.0)
        ax23.set_aspect(1.0)
        f2.tight_layout()
        f2.subplots_adjust(top=0.8)
        
        

        plt.show()
   
   
    def test3DInterpolation(self):
        def func(x,y,z):
            r = np.sqrt(x**2+y**2 + z**2)
#             return np.exp(-np.sqrt(x**2+y**2))
#             return np.exp(-r)/(r) 
#             return np.exp(-r) 
            return r**2 

#         def func(x,y,xt,yt):
#             r = np.sqrt((x-xt)**2+(y-yt)**2)
#             return np.exp(-r)/(r) 

#         title = r"$f(x,y) = exp(-\sqrt{x^2+y^2})$"
#         title = r"$f(r) = exp(-r)/r$"
        title = r"$f(r) = exp(-r)$"

        npoints = 20

        xmin = 0
        ymin = 0
        zmin = 0
        xmax =  1
        ymax =  1
        zmax =  1
        testXpoints = np.linspace(xmin,xmax,npoints)
        testYpoints = np.linspace(ymin,ymax,npoints)
        testZpoints = np.linspace(zmin,zmax,npoints)
        testf = np.zeros((npoints,npoints,npoints))
        for i in range(npoints):
            for j in range(npoints):
                for k in range(npoints):
                    testf[i,j,k] = func(testXpoints[i],testYpoints[j], testZpoints[k])
                
        nChebPtsA = 5
        
        
        
        xCheb = ChebyshevPoints(xmin, xmax, nChebPtsA)
        yCheb = ChebyshevPoints(ymin, ymax, nChebPtsA)
        zCheb = ChebyshevPoints(zmin, zmax, nChebPtsA)
        xChebMesh1A, yChebMesh1A, zChebMesh1A = np.meshgrid(xCheb,yCheb,zCheb)

        fCheb = np.zeros((nChebPtsA,nChebPtsA,nChebPtsA)) 
        for i in range(nChebPtsA):
            for j in range(nChebPtsA):
                for k in range(nChebPtsA):
                    fCheb[i,j,k] = func(xCheb[i],yCheb[j],zCheb[k])      
        P1A = interpolator3Dchebyshev(xCheb, yCheb, zCheb, fCheb)
        
        
       
        
        xg1, yg1, zg1 = np.meshgrid(testXpoints, testYpoints,  testZpoints, indexing='ij')
        interpResult1A = P1A(xg1, yg1, zg1)
        
        
        diff = np.abs(interpResult1A-testf)
        print(interpResult1A[0,0,0])
#         print(interpResult1A)
        print(testf[0,0,0])
#         print(testf)
#         print(diff[0,0,0])
#         print('shape: ', np.shape(diff.flatten()))
#         print('shape: ', np.shape(xg1.flatten()))
        idx = np.argmax(diff.flatten())
        print('Max error: ', diff.flatten()[idx])
        print('Occured at: ', xg1.flatten()[idx], yg1.flatten()[idx], zg1.flatten()[idx])
        

      
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()