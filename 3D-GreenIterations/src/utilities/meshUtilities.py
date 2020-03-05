'''
Mesh utilities for the adaptive mesh refinement.

@author: nathanvaughn
'''
from numpy import pi, cos, arccos, sin, sqrt, exp, abs
import numpy as np
from scipy.special import factorial, comb, erf
import vtk
import time

import interpolation_wrapper


def interapolateBetweenTwoMeshesSingleCell(coarseX, coarseY, coarseZ, coarseF,
                                fineX, fineY, fineZ,
                                wx,wy,wz):
    
    start=time.time()
    ptsPerDim=len(wx)
    uniqueX = coarseX[::ptsPerDim*ptsPerDim]
    uniqueY = coarseY[0:ptsPerDim*ptsPerDim:ptsPerDim]
    uniqueZ = coarseZ[0:ptsPerDim]
    
    # create interpolator
    def P3(xt,yt,zt):  # 3D interpolator.  
        
        num = 0
        idx=0
        for i in range(len(wx)):
            numY = 0
            for j in range(len(wy)):
                numZ = 0
                for k in range(len(wz)):
                    numZ += ( wz[k]/(zt-uniqueZ[k])*coarseF[idx] )
                    idx+=1
#                     print(idx)
                    
                numY += ( wy[j]/(yt-uniqueY[j]) )*numZ
            num +=  ( wx[i]/(xt-uniqueX[i]) )*numY
        
        denX=0
        # issue is that coarseX isn't just the unique values of X, it's all of them.  
        for i in range(len(wx)):
            denX += wx[i]/(xt-uniqueX[i])
        
        denY=0
        for j in range(len(wy)):
            denY += wy[j]/(yt-uniqueY[j])
            
        denZ=0
        for k in range(len(wz)):
            denZ += wz[k]/(zt-uniqueZ[k])
        
        den = denX*denY*denZ
        return num/den
    
    # vectorize interpolator
    interpolator = np.vectorize(P3)
    end=time.time()
    print("Time to vectorize = ", end-start)
    # call interpolator
    fineF = interpolator(fineX,fineY,fineZ)
    print("Time to use vectorized func = ", time.time()-end)
    print()
    
    return fineF

def interapolateBetweenTwoMeshesSingleCellNoVector(coarseX, coarseY, coarseZ, coarseF,
                                fineX, fineY, fineZ,
                                wx,wy,wz):
    
#     start=time.time()
    ptsPerDim=len(wx)
    uniqueX = coarseX[::ptsPerDim*ptsPerDim]
    uniqueY = coarseY[0:ptsPerDim*ptsPerDim:ptsPerDim]
    uniqueZ = coarseZ[0:ptsPerDim]
    
    num = 0
    idx=0
    for i in range(len(wx)):
        numY = 0
        for j in range(len(wy)):
            numZ = 0
            for k in range(len(wz)):
                numZ += ( wz[k]/(fineZ-uniqueZ[k])*coarseF[idx] )
                idx+=1
                
            numY += ( wy[j]/(fineY-uniqueY[j]) )*numZ
        num +=  ( wx[i]/(fineX-uniqueX[i]) )*numY
    
    denX=0
    
    for i in range(len(wx)):
        denX += wx[i]/(fineX-uniqueX[i])
    
    denY=0
    for j in range(len(wy)):
        denY += wy[j]/(fineY-uniqueY[j])
        
    denZ=0
    for k in range(len(wz)):
        denZ += wz[k]/(fineZ-uniqueZ[k])
    
    den = denX*denY*denZ
    
    
    trueFineF=1+fineX**4 * fineY**4
#     print("\nTrue fineF = ", trueFineF)
#     print("\nInterpolated fineF = ", num/den)
#     print("\nFine Error = ", (num/den-trueFineF)/trueFineF)

    
#     end=time.time()
#     print("Time without vectorizing: ", end-start)
#     print()
    return num/den
    

def PROTOTYPE_FOR_SINGULARITY_CATCHING__interapolateBetweenTwoMeshesSingleCellNoVector(coarseX, coarseY, coarseZ, coarseF,
                                fineX, fineY, fineZ,
                                wx,wy,wz):
    
#     start=time.time()
    ptsPerDim=len(wx)
    uniqueX = coarseX[::ptsPerDim*ptsPerDim]
    uniqueY = coarseY[0:ptsPerDim*ptsPerDim:ptsPerDim]
    uniqueZ = coarseZ[0:ptsPerDim]
    
    print("Coarse F = ", coarseF)
    
    # create interpolator
#     def P3(xt,yt,zt):  # 3D interpolator.  
        
    num = 0
    idx=0

    
    x_exact=-np.ones(len(fineX),dtype=int)
    y_exact=-np.ones(len(fineY),dtype=int)
    z_exact=-np.ones(len(fineZ),dtype=int)
    
    numYs = np.zeros((len(wx),len(fineX)))
    for i in range(len(wx)):
        print("i=",i)
        dx=fineX-uniqueX[i]
        
        x_exact = np.where(abs(dx)>1e-14, x_exact, i) # y_exact stores the index for which y_sum should be used
        
        numY = 0
#         y_exact=-np.ones(len(fineY),dtype=int)
        numZs = np.zeros( (len(wy),len(fineY)) )
        for j in range(len(wy)):
            print("\tj=",j)
            dy=fineY-uniqueY[j]
            
            y_exact = np.where(abs(dy)>1e-14, y_exact, j)  # y_exact stores the index for which z_sum should be used
            
            numZ = 0
            
            
            for k in range(len(wz)):
                print("\t\tk=",k)
#                 print("z=",uniqueZ[k])
#                 print(np.shape(fineZ))
                dz=fineZ-uniqueZ[k]
#                 z_exact_old = np.copy(z_exact)
                
                
                z_exact = np.where(abs(dz)>1e-14, z_exact, idx)  # z_exact stores the index of the point in coarseF
                print("\t\t\tidx     ",idx)
#                 print("\t\t\tx_exact",x_exact)
#                 print("\t\t\ty_exact",y_exact)
#                 print("\t\t\tz_exact",z_exact)

                numZ = np.where((z_exact==-1) ,   numZ +wz[k]/(dz)*coarseF[idx], coarseF[z_exact])
                

                idx+=1
            

            numZs[j,:] = numZ
#             print("np.shape(numY) ", np.shape(numY))
#             print("np.shape(numZ) ", np.shape(numZ))
#             print("np.shape(numZs) ", np.shape(numZs))
#             print("slice: numZs[y_exact,:] ", np.shape( numZs[max(y_exact),:]) )
            numY =  np.where(y_exact==-1, (numY + wy[j]/dy)*numZ, numZs[max(y_exact),:])
#             print("np.shape(numY) ", np.shape(numY))

        print(np.shape(numY))
        print(np.shape(numYs))
        numYs[i,:] = numY
        num = np.where(x_exact==-1, (num +  wx[i]/dx)*numY, numYs[max(x_exact),:])
        
#         print("\tnum = ",num)


    denX=0

    for i in range(len(wx)):
        dx=(fineX-uniqueX[i])
        dx=np.where( abs(dx)>1e-14, dx, 1)
        denX += wx[i]/dx

    denX = np.where(x_exact==-1,denX,1)

    denY=0
    for j in range(len(wy)):
        dy=(fineY-uniqueY[j])
        dy=np.where( abs(dy)>1e-14, dy, 1) 
        denY += wy[j]/dy
        
    denY = np.where(y_exact==-1,denY,1)

        
    denZ=0
    for k in range(len(wz)):
        dz=(fineZ-uniqueZ[k])
        dz=np.where( abs(dz)>1e-14, dz, 1)
        denZ += wz[k]/dz
    
    denZ = np.where(z_exact==-1,denZ,1)
#     print(denZ)
    
    den = denX*denY*denZ
    
    
#     print(numZ)
#     print(numY)
#     print(num)
    
    
#     print(num[int(len(den)/2)])
#     print(den[int(len(den)/2)])
#     print(coarseF[int(len(coarseF)/2)])

#     print()
#     print(np.sqrt(coarseX**2 + coarseY**2 + coarseZ**2))
#     print(coarseF)
#     print(np.sqrt(fineX**2 + fineY**2 + fineZ**2))
    
    trueFineF=fineX**4 * fineY**4
#     print("\nTrue fineF = ", trueFineF)
#     print("\nInterpolated fineF = ", num/den)
#     print("Coarse Error = ", coarseF-(coarseX**4 * coarseY**4))
#     print("\nInterpolated field Error = ", (num/den-trueFineF)/trueFineF)
    
    
    
    
#     end=time.time()
#     print("Time without vectorizing: ", end-start)
#     print()
    return num/den



def PROTOTYPE2_FOR_SINGULARITY_CATCHING__interapolateBetweenTwoMeshesSingleCellNoVector(coarseX, coarseY, coarseZ, coarseF,
                                fineX, fineY, fineZ,
                                wx,wy,wz, 
                                singularities=[]):
    
#     start=time.time()
    ptsPerDim=len(wx)
    uniqueX = coarseX[::ptsPerDim*ptsPerDim]
    uniqueY = coarseY[0:ptsPerDim*ptsPerDim:ptsPerDim]
    uniqueZ = coarseZ[0:ptsPerDim]
    
    num = 0
    idx=0
    for i in range(len(wx)):
        numY = 0
        for j in range(len(wy)):
            numZ = 0
            for k in range(len(wz)):
                dz=(fineZ-uniqueZ[k])
                dz=np.where( abs(dz)>1e-14, dz, 1)
                numZ += ( wz[k]/dz*coarseF[idx] )
                idx+=1
             
            dy=(fineY-uniqueY[j])
            dy=np.where( abs(dy)>1e-14, dy, 1)   
            numY += ( wy[j]/dy )*numZ
        dx=(fineX-uniqueX[i])
        dx=np.where( abs(dx)>1e-14, dx, 1)   
        num +=  ( wx[i]/dx )*numY
    
    denX=0
    
    for i in range(len(wx)):
        dx=(fineX-uniqueX[i])
        dx=np.where( abs(dx)>1e-14, dx, 1)
        denX += wx[i]/dx
    
    denY=0
    for j in range(len(wy)):
        dy=(fineY-uniqueY[j])
        dy=np.where( abs(dy)>1e-14, dy, 1) 
        denY += wy[j]/dy
        
    denZ=0
    for k in range(len(wz)):
        dz=(fineZ-uniqueZ[k])
        dz=np.where( abs(dz)>1e-14, dz, 1)
        denZ += wz[k]/dz
    
    den = denX*denY*denZ
    
    
    print("Singularities inside: ", singularities)
    for singularity in singularities:
        
        coarseIdx=singularity[0]
        fineIdx=singularity[1]
        print("Fixing singularity at x,y,z = ", coarseX[coarseIdx], coarseY[coarseIdx], coarseZ[coarseIdx])
        
        den[fineIdx]=1
        num[fineIdx]=coarseF[coarseIdx]
        
    
    trueFineF=1+fineX**4 * fineY**4
#     print("\nTrue fineF = ", trueFineF)
#     print("\nInterpolated fineF = ", num/den)
#     print("\nFine Error = ", (num/den-trueFineF)/trueFineF)
    
    

    
#     end=time.time()
#     print("Time without vectorizing: ", end-start)
#     print()
    return num/den

def interpolateBetweenTwoMeshes(coarseX, coarseY, coarseZ, coarseF, pointsPerCell_coarse,
                                fineX, fineY, fineZ, pointsPerCell_fine):
    '''
    Interpolates between two meshes.  
    Hard assumption that the meshes are arranged the same way, so that the first (p+1)**3 points of the coarse mesh correspond to the 
    same cell as the first (P+1)**3 points of the fine mesh.  
    '''
    
#     start=time.time()
#     pointsPerCoarseCell=(coarseP+1)**3
#     pointsPerFineCell=(fineP+1)**3    
#     # get the number of cells
#     numCellsCoarse = int(len(coarseX)/pointsPerCoarseCell)
#     numCellsFine = int(len(fineX)/pointsPerFineCell)
#     
# #     print("pointsPerCoarseCell ", pointsPerCoarseCell)
# #     print("pointsPerFineCell ", pointsPerFineCell)
# #     print("numCellsCoarse ", numCellsCoarse)
# #     print("numCellsFine ", numCellsFine)
#     assert numCellsCoarse==numCellsFine, "numCellsCoarse != numCellsFine"
    
    if len(coarseX)==len(fineX):  # meshes are the same, don't need to interpolate.
        return coarseF
    
#     print(pointsPerCell_coarse)
    pointsPerCell_coarse = pointsPerCell_coarse.astype(int)
    pointsPerCell_fine = pointsPerCell_fine.astype(int)
    
    
#     numCells = numCellsFine
    
    fineF=np.zeros(len(fineX))
    coarseP = int(np.cbrt(pointsPerCell_coarse[0]))-1
    
#     print(pointsPerCell_coarse)
#     print(courseP)
#     exit(-1)
    # Set up weight arrays.
    wx = np.ones(coarseP+1)
    for i in range(coarseP+1):
        wx[i] = (-1)**i * np.sin(  (2*i+1)*np.pi / (2*(coarseP)+2)  )
    
    wy = np.ones(coarseP+1)
    for j in range(coarseP+1):
        wy[j] = (-1)**j * np.sin(  (2*j+1)*np.pi / (2*(coarseP)+2)  )
        
    wz = np.ones(coarseP+1)
    for k in range(coarseP+1):
        wz[k] = (-1)**k * np.sin(  (2*k+1)*np.pi / (2*(coarseP)+2)  )
    
    
    
    # call the single-cell function for each cell
    coarseIdx=0
    fineIdx = 0
    for i in range(len(pointsPerCell_coarse)):
        
        if pointsPerCell_coarse[i]==pointsPerCell_fine[i]:
            fineF[fineIdx:fineIdx+pointsPerCell_fine[i]] = coarseF[coarseIdx:coarseIdx+pointsPerCell_coarse[i]]
        else:
            
#             print("Interpolating for a cell containing %i and %i points" %(pointsPerCell_coarse[i],pointsPerCell_fine[i]))
#          fineF[i*pointsPerFineCell:(i+1)*pointsPerFineCell] = interapolateBetweenTwoMeshesSingleCellNoVector(
            fineF[fineIdx:fineIdx+pointsPerCell_fine[i]] = interapolateBetweenTwoMeshesSingleCellNoVector(
                                                               coarseX[coarseIdx:coarseIdx+pointsPerCell_coarse[i]],
                                                               coarseY[coarseIdx:coarseIdx+pointsPerCell_coarse[i]],
                                                               coarseZ[coarseIdx:coarseIdx+pointsPerCell_coarse[i]],
                                                               coarseF[coarseIdx:coarseIdx+pointsPerCell_coarse[i]],
                                                               fineX[fineIdx:fineIdx+pointsPerCell_fine[i]],
                                                               fineY[fineIdx:fineIdx+pointsPerCell_fine[i]],
                                                               fineZ[fineIdx:fineIdx+pointsPerCell_fine[i]],
                                                               wx,wy,wz)
        coarseIdx += pointsPerCell_coarse[i]
        fineIdx += pointsPerCell_fine[i]

#     end=time.time()
#     print("Interpolating between meshes took %2.3f seconds " %(end-start))
#     print("returning fineF of length ", len(fineF))
    return fineF


def interpolateBetweenTwoMeshes_variableOrder(coarseX, coarseY, coarseZ, coarseF, coarseP,
                                fineX, fineY, fineZ, fineP, singularities=[]):
    '''
    Interpolates between two meshes.  
    Hard assumption that the meshes are arranged the same way, so that the first (p+1)**3 points of the coarse mesh correspond to the 
    same cell as the first (P+1)**3 points of the fine mesh.  
    '''
    
#     start=time.time()
    pointsPerCoarseCell=(coarseP+1)**3
    pointsPerFineCell=(fineP+1)**3    
    # get the number of cells
    numCellsCoarse = int(len(coarseX)/pointsPerCoarseCell)
    numCellsFine = int(len(fineX)/pointsPerFineCell)
    
#     print("pointsPerCoarseCell ", pointsPerCoarseCell)
#     print("pointsPerFineCell ", pointsPerFineCell)
#     print("numCellsCoarse ", numCellsCoarse)
#     print("numCellsFine ", numCellsFine)
    assert numCellsCoarse==numCellsFine, "numCellsCoarse != numCellsFine"
    
    if len(coarseX)==len(fineX):  # meshes are the same, don't need to interpolate.
        return coarseF
    
    
    numCells = numCellsFine
    
    fineF=np.zeros(len(fineX))
    
    # Set up weight arrays.
    wx = np.ones(coarseP+1)
    for i in range(coarseP+1):
        wx[i] = (-1)**i * np.sin(  (2*i+1)*np.pi / (2*(coarseP)+2)  )
    
    wy = np.ones(coarseP+1)
    for j in range(coarseP+1):
        wy[j] = (-1)**j * np.sin(  (2*j+1)*np.pi / (2*(coarseP)+2)  )
        
    wz = np.ones(coarseP+1)
    for k in range(coarseP+1):
        wz[k] = (-1)**k * np.sin(  (2*k+1)*np.pi / (2*(coarseP)+2)  )
    
    # call the single-cell function for each cell
    for i in range(numCells):
#        fineF[i*pointsPerFineCell:(i+1)*pointsPerFineCell] = PROTOTYPE_FOR_SINGULARITY_CATCHING__interapolateBetweenTwoMeshesSingleCellNoVector(
         fineF[i*pointsPerFineCell:(i+1)*pointsPerFineCell] = interapolateBetweenTwoMeshesSingleCellNoVector(
                                                               coarseX[i*pointsPerCoarseCell: (i+1)*pointsPerCoarseCell],
                                                               coarseY[i*pointsPerCoarseCell: (i+1)*pointsPerCoarseCell],
                                                               coarseZ[i*pointsPerCoarseCell: (i+1)*pointsPerCoarseCell],
                                                               coarseF[i*pointsPerCoarseCell: (i+1)*pointsPerCoarseCell],
                                                               fineX[i*pointsPerFineCell: (i+1)*pointsPerFineCell],
                                                               fineY[i*pointsPerFineCell: (i+1)*pointsPerFineCell],
                                                               fineZ[i*pointsPerFineCell: (i+1)*pointsPerFineCell],
                                                               wx,wy,wz)

#     end=time.time()
#     print("Interpolating between meshes took %2.3f seconds " %(end-start))
#     print("returning fineF of length ", len(fineF))
    return fineF



def meshDensity(r,divideParameter,divideCriterion):
    '''
    Mesh density function from Wilkinson and Levine for order 2, total gridpoints roughly N
    :param N:
    :param r:
    '''
    
    
    if divideCriterion == 'LW1':
        # for order = 1
        return divideParameter/25.191*(exp(-2*r)* (4 - 2/r + 9/r**2) )**(3/5)
    
    elif divideCriterion == 'LW2':
        # for order = 2 
        return divideParameter*2/412.86*(exp(-2*r)* (64 - 78/r + 267/r**2 + 690/r**3 + 345/r**4) )**(3/7)
    
    elif divideCriterion == 'LW3':
        # for order = 3 
        return divideParameter/648.82*(exp(-2*r)* (52 - 102/r + 363/r**2 + 1416/r**3 + 4164/r**4 + 5184/r**5 + 2592/r**6) )**(3/9)
    
    elif divideCriterion == 'LW4':
        # for order = 3 
        return divideParameter/1798*(exp(-2*r)* (423 - 1286/r + 2875/r**2 + 16506/r**3 + 79293/r**4 + 292512/r**5 + 611136/r**6
                                                 + 697320/r**7 + 348660/r**8) )**(3/11)
    
    elif divideCriterion == 'LW5':
        # for order = 3 
        return divideParameter/3697.1*(exp(-2*r)* (2224 - 9018/r + 16789/r**2 + 117740/r**3 + 733430/r**4 + 3917040/r**5 + 16879920/r**6
                                                   + 49186500/r**7 + 91604250/r**8 + 100516500/r**9 + 50258250/r**10) )**(3/13)
    
    elif divideCriterion == 'LW3_modified':
        # for order = 3 
        return divideParameter/20*(exp(-8*r)* (52 - 102/r + 363/r**2 + 1416/r**3 + 4164/r**4 + 5184/r**5 + 2592/r**6) )**(3/9)
    
    elif divideCriterion == 'Phani':
        N = 8
        eta = np.sqrt(2*0.34)
        k = 5
        return phaniMeshDensity(divideParameter, N, eta,k, r)
    
    elif divideCriterion == 'Krasny_density':
#         return divideParameter/3697.1*50258250**(3/13)*(np.exp(-6*r/13)/r**(30/13))
                                                   
        k = np.sqrt(2*0.1)
#         k = np.sqrt(2*0.2)
#         return divideParameter/3697.1*50258250**(3/13)*(np.exp(-k*r)/r**(30/13))
        return divideParameter/3697.1*exp(-k*r)* (2224 - 9018/r + 16789/r**2 + 117740/r**3 + 733430/r**4 + 3917040/r**5 + 16879920/r**6
                   + 49186500/r**7 + 91604250/r**8 + 100516500/r**9 + 50258250/r**10) **(3/13)

#         k=2
#         return divideParameter*(np.exp(-2*r))*(1/r**2)
#         return divideParameter*(np.exp(-k*r))*(1+1/r**2)
#         return divideParameter*(np.exp(-1*k*r))*(k+1/r**1)

#         return divideParameter*1/r**(2)
#         return divideParameter*(np.exp(-1*r))/r**2

#         return divideParameter*(np.exp(-0.3*r))/r**2
#         return ( 1/r + divideParameter*np.exp(-0.5*r) )
#         return divideParameter/r
    
    
    elif divideCriterion == 'Nathan_density':
        
        rho0 = 308.88
        rhoC = 1.085
        rC = 0.5
        k1 = 8
        k2 = 0.824
        
        
        
        if r < rC:
            return (k1*divideParameter* ( np.log(rho0)-np.log(rhoC) )/ np.log(rho0)**2)**3
        else:
            return (k2*divideParameter / np.log(rho0) )**3
   
   
#         if r < rC:
#             return rho0*k1*divideParameter *np.exp(-k1*r)
#         else:
#             return  (rho0*k1*divideParameter *np.exp(-k1*rC)  /  (rhoC*k2*divideParameter *np.exp( -k2*(rC-rC) ))  )  * rhoC*k2*divideParameter *np.exp( -k2*(r-rC) )
        
    else:
        print('Invalid Mesh type...')
        return
    
def phaniMeshDensity(A, N,eta,k,r):
    innersum = 0
    for n in range(k+2): #sum from 0, through k+1
        innersum += comb(k+1,n) * 2**n * eta**n * factorial(k+1-n) / r**(k-n+2)
        
        
    h = A * (  N/pi * eta**(2*k+5) * exp(-2*eta*r)  +  N**2*exp(-4*eta*r) * (
        eta**(k+2) * 2**(k+1) * innersum 
        ) **2
        ) ** (-1/(2*k+3))
#     print(h)
    return h**(-3)

    
def computeCoefficicents(f):
    (px,py,pz) = np.shape(f)
    print(px,py,pz)
    x = ChebyshevPointsFirstKind(-1, 1, px)
    y = ChebyshevPointsFirstKind(-1, 1, py)
    z = ChebyshevPointsFirstKind(-1, 1, pz)
    
    coefficients = np.zeros_like(f)
    
    # Loop through coefficients
    for i in range(px):
#         print()
#         print()
        for j in range(py):
#             print()
            for k in range(pz):
                
                # Loop through quadrature sums
                
                for ell in range(px):
#                     theta_ell = (ell+1/2)*pi/px
                    theta_ell = arccos(x[ell])
                    for m in range(py):
#                         theta_m = (m+1/2)*pi/py
                        theta_m = arccos(y[m])
                        for n in range(pz):
#                             theta_n = (n+1/2)*pi/pz
                            theta_n = arccos(z[n])
                            
                            coefficients[i,j,k] += f[ell,m,n] * cos(i*theta_ell) * cos(j*theta_m) * cos(k*theta_n)
#                             coefficients[i,j,k] += f[ell,m,n] * cos(i*x[ell]) * cos(j*y[m]) * cos(k*z[n])
                
                
                coefficients[i,j,k] *= (2/px)*(2/py)*(2/pz)
                
                if i==0:
                    coefficients[i,j,k]/=2
                if j==0:
                    coefficients[i,j,k]/=2
                if k==0:
                    coefficients[i,j,k]/=2
                    
    
                if abs(coefficients[i,j,k]) > 1e-12:
                    print('alpha(%i,%i,%i) = %e' %(i,j,k,coefficients[i,j,k]) )
    
    return coefficients

def sumChebyshevCoefficicentsGreaterThanOrderQ(f,q):
#     print('\n q = ', q, '\n')
    (px,py,pz) = np.shape(f)
#     print(px,py,pz)
    x = ChebyshevPointsFirstKind(-1, 1, px)
    y = ChebyshevPointsFirstKind(-1, 1, py)
    z = ChebyshevPointsFirstKind(-1, 1, pz)
    
    coefficients = np.zeros_like(f)
    
    # Loop through coefficients
    for i in range(px):
#         print()
#         print()
        for j in range(py):
#             print()
            for k in range(pz):
                
                if (i+j+k)>=q:
                    # Loop through quadrature sums
                    
                    for ell in range(px):
    #                     theta_ell = (ell+1/2)*pi/px
                        theta_ell = arccos(x[ell])
                        for m in range(py):
    #                         theta_m = (m+1/2)*pi/py
                            theta_m = arccos(y[m])
                            for n in range(pz):
    #                             theta_n = (n+1/2)*pi/pz
                                theta_n = arccos(z[n])
                                
                                coefficients[i,j,k] += f[ell,m,n] * cos(i*theta_ell) * cos(j*theta_m) * cos(k*theta_n)
    #                             coefficients[i,j,k] += f[ell,m,n] * cos(i*x[ell]) * cos(j*y[m]) * cos(k*z[n])
    
#                 if abs(coefficients[i,j,k]) > 1e-12:
#                     print('alpha(%i,%i,%i) = %e' %(i,j,k,coefficients[i,j,k]) )
    
                
                
                    coefficients[i,j,k] *= (2/px)*(2/py)*(2/pz)
                    
                    if i==0:
                        coefficients[i,j,k]/=2
                    if j==0:
                        coefficients[i,j,k]/=2
                    if k==0:
                        coefficients[i,j,k]/=2
                        
        
#                     if abs(coefficients[i,j,k]) > 1e-12:
#                         print('alpha(%i,%i,%i) = %e' %(i,j,k,coefficients[i,j,k]) )
    
#     if q==12: 
#         print()
#         print(coefficients)
#         print()
    return np.sum(np.abs(coefficients))

def sumChebyshevCoefficicentsEachGreaterThanOrderQ(f,q):
#     print('\n q = ', q, '\n')
    (px,py,pz) = np.shape(f)
#     print(px,py,pz)
    x = ChebyshevPointsFirstKind(-1, 1, px)
    y = ChebyshevPointsFirstKind(-1, 1, py)
    z = ChebyshevPointsFirstKind(-1, 1, pz)
    
    coefficients = np.zeros_like(f)
    
    # Loop through coefficients
    for i in range(px):
#         print()
#         print()
        for j in range(py):
#             print()
            for k in range(pz):
                
#                 if (i+j+k)>=q:
                if ( (i>=q) and (j>=q) and (k>=q) ):
                    # Loop through quadrature sums
                    
                    for ell in range(px):
    #                     theta_ell = (ell+1/2)*pi/px
                        theta_ell = arccos(x[ell])
                        for m in range(py):
    #                         theta_m = (m+1/2)*pi/py
                            theta_m = arccos(y[m])
                            for n in range(pz):
    #                             theta_n = (n+1/2)*pi/pz
                                theta_n = arccos(z[n])
                                
                                coefficients[i,j,k] += f[ell,m,n] * cos(i*theta_ell) * cos(j*theta_m) * cos(k*theta_n)
    #                             coefficients[i,j,k] += f[ell,m,n] * cos(i*x[ell]) * cos(j*y[m]) * cos(k*z[n])
    
#                 if abs(coefficients[i,j,k]) > 1e-12:
#                     print('alpha(%i,%i,%i) = %e' %(i,j,k,coefficients[i,j,k]) )
    
                
                
                    coefficients[i,j,k] *= (2/px)*(2/py)*(2/pz)
                    
                    if i==0:
                        coefficients[i,j,k]/=2
                    if j==0:
                        coefficients[i,j,k]/=2
                    if k==0:
                        coefficients[i,j,k]/=2
                        
        
#                     if abs(coefficients[i,j,k]) > 1e-12:
#                         print('alpha(%i,%i,%i) = %e' %(i,j,k,coefficients[i,j,k]) )
    
#     if q==12: 
#         print()
#         print(coefficients)
#         print()
    return np.sum(np.abs(coefficients))

def sumChebyshevCoefficicentsAnyGreaterThanOrderQ(f,q):
#     print('\n q = ', q, '\n')
    (px,py,pz) = np.shape(f)
#     print(px,py,pz)
    x = ChebyshevPointsFirstKind(-1, 1, px)
    y = ChebyshevPointsFirstKind(-1, 1, py)
    z = ChebyshevPointsFirstKind(-1, 1, pz)
    
    coefficients = np.zeros_like(f)
    
    # Loop through coefficients
    for i in range(px):
#         print()
#         print()
        for j in range(py):
#             print()
            for k in range(pz):
                
#                 if (i+j+k)>=q:
                if ( (i>=q) or (j>=q) or (k>=q) ):
                    # Loop through quadrature sums
                    
                    for ell in range(px):
    #                     theta_ell = (ell+1/2)*pi/px
                        theta_ell = arccos(x[ell])
                        for m in range(py):
    #                         theta_m = (m+1/2)*pi/py
                            theta_m = arccos(y[m])
                            for n in range(pz):
    #                             theta_n = (n+1/2)*pi/pz
                                theta_n = arccos(z[n])
                                
                                coefficients[i,j,k] += f[ell,m,n] * cos(i*theta_ell) * cos(j*theta_m) * cos(k*theta_n)
    #                             coefficients[i,j,k] += f[ell,m,n] * cos(i*x[ell]) * cos(j*y[m]) * cos(k*z[n])
    
#                 if abs(coefficients[i,j,k]) > 1e-12:
#                     print('alpha(%i,%i,%i) = %e' %(i,j,k,coefficients[i,j,k]) )
    
                
                
                    coefficients[i,j,k] *= (2/px)*(2/py)*(2/pz)
                    
                    if i==0:
                        coefficients[i,j,k]/=2
                    if j==0:
                        coefficients[i,j,k]/=2
                    if k==0:
                        coefficients[i,j,k]/=2

    return np.sum(np.abs(coefficients))

def sumChebyshevCoefficicentsGreaterThanOrderQZeroZero(f,q):
#     print('\n q = ', q, '\n')
    (px,py,pz) = np.shape(f)
#     print(px,py,pz)
    x = ChebyshevPointsFirstKind(-1, 1, px)
    y = ChebyshevPointsFirstKind(-1, 1, py)
    z = ChebyshevPointsFirstKind(-1, 1, pz)
    
    coefficients = np.zeros_like(f)
    
    # Loop through coefficients
#     for i in range(px):
#         for j in range(py):
#             for k in range(pz):
                
#                 if ( (i>=q) or (j>=q) or (k>=q) ):
#                     # Loop through quadrature sums
                    
    for ell in range(px):
        theta_ell = arccos(x[ell])
        for m in range(py):
            theta_m = arccos(y[m])
            for n in range(pz):
                theta_n = arccos(z[n])
                
                coefficients[q,0,0] += f[ell,m,n] * cos(q*theta_ell) * cos(0*theta_m) * cos(0*theta_n)
                coefficients[0,q,0] += f[ell,m,n] * cos(0*theta_ell) * cos(q*theta_m) * cos(0*theta_n)
                coefficients[0,0,q] += f[ell,m,n] * cos(0*theta_ell) * cos(0*theta_m) * cos(q*theta_n)
    
#                 if abs(coefficients[i,j,k]) > 1e-12:
#                     print('alpha(%i,%i,%i) = %e' %(i,j,k,coefficients[i,j,k]) )
    
                
                
    coefficients[q,0,0] *= (2/px)*(2/py)*(2/pz) / 4
    coefficients[0,q,0] *= (2/px)*(2/py)*(2/pz) / 4
    coefficients[0,0,q] *= (2/px)*(2/py)*(2/pz) / 4
                    
                        

    return np.sum(np.abs(coefficients))
    
                
                
            

def unscaledWeightsFirstKind(N):
    # generate Lambda
    Lambda = np.ones((N+1,N+1))
    for i in range(N+1):
        for j in range(N+1):
            j_shift = j+1/2
            Lambda[i,j] = 2/(N+1) * cos(i*j_shift*pi/(N+1))

    W = np.zeros(N+1)
    for i in range(N+1):
        if i == 0:
            W[i] = 1
        elif i%2==0:
            W[i] = 2/(1-i**2)
        else:
            W[i] = 0
            
    w = np.dot(np.transpose(Lambda),W)
    return w

def weightsFirstKind(xlow, xhigh, N, w=None):
#     if w != None:
    try: 
        return (xhigh - xlow)/2 * w
    except TypeError:
#         print('meshUtilities: Generating weights from scratch')
        return (xhigh - xlow)/2 *unscaledWeightsFirstKind(N)
    
def weights3DFirstKind(xlow,xhigh,Nx,ylow,yhigh,Ny,zlow,zhigh,Nz,w=None):
    xw = weightsFirstKind(xlow, xhigh, Nx, w)
    yw = weightsFirstKind(ylow, yhigh, Ny, w)
    zw = weightsFirstKind(zlow, zhigh, Nz, w)
    
    return np.outer( np.outer(xw,yw), zw ).reshape([Nx+1,Ny+1,Nz+1])
        
def ChebyshevPointsFirstKind(xlow, xhigh, N):
    '''
    Generates "open" Chebyshev points. N midpoints in theta.
    '''
    endpoints = np.linspace(np.pi,0,N+2)
    theta = (endpoints[1:] + endpoints[:-1])/2
#     print(theta)
    u = np.cos(theta)
    x = xlow + (xhigh-xlow)/2*(u+1)
    return x

def unscaledWeightsSecondKind(N):
    # generate Lambda
    Lambda = np.ones((N+1,N+1))
    for i in range(N+1):
        for j in range(N+1):
#             j_shift = j+1/2
            if ( (j==0) or (j==N) ):
                Lambda[i,j] = 1/N * cos(i*(j)*pi/N)
            else:
                Lambda[i,j] = 2/N * cos(i*(j)*pi/N)

    W = np.zeros(N+1)
    for i in range(N+1):
        if i == 0:
            W[i] = 1
        elif i%2==0:
            W[i] = 2/(1-i**2)
        else:
            W[i] = 0
        if (N)%2==0:  # this seems necessary for even nx,ny,nz
            W[-1] = 1/(1-N**2)
            
    w = np.dot(np.transpose(Lambda),W)
    return w

def weightsSecondKind(xlow, xhigh, N, w=None):
#     if w != None:
    try: 
        return (xhigh - xlow)/2 * w
    except TypeError:
#         print('meshUtilities: Generating weights from scratch')
        return (xhigh - xlow)/2 *unscaledWeightsSecondKind(N)
    
def weights3DSecondKind(xlow,xhigh,Nx,ylow,yhigh,Ny,zlow,zhigh,Nz,w=None):
    xw = weightsSecondKind(xlow, xhigh, Nx, w)
    yw = weightsSecondKind(ylow, yhigh, Ny, w)
    zw = weightsSecondKind(zlow, zhigh, Nz, w)
    
    return np.outer( np.outer(xw,yw), zw ).reshape([Nx+1,Ny+1,Nz+1])
        
def ChebyshevPointsSecondKind(xlow, xhigh, N):
    '''
    Generates "open" Chebyshev points. N midpoints in theta.
    '''
#     endpoints = np.linspace(np.pi,0,N+1)
    theta = np.linspace(np.pi,0,N+1)
#     print(theta)
    u = np.cos(theta)
    x = xlow + (xhigh-xlow)/2*(u+1)
    return x


def mapToMinusOneToOne(xlow, xhigh, x):
    '''
    Generates "open" Chebyshev points. N midpoints in theta.
    '''
#     N = np.len(x)
#     endpoints = np.linspace(np.pi,0,N+1)
#     theta = (endpoints[1:] + endpoints[:-1])/2
# #     print(theta)
#     u = np.cos(theta)
#     x = xlow + (xhigh-xlow)/2*(u+1)
#     return x
    x = x - (xhigh+xlow)/2  # shifts center to 0
    print(x)
    x = x*2/(xhigh-xlow) # scales x to [-1,1]
    return x

def Tprime(n,x):
    output = np.empty_like(x)
    for i in range(output.size):
        if x[i] == 1:
            print('x[i] == 1')
            output[i] = n**2
        elif x[i] == -1:
            print('x[i] == -1')
            output[i] = (-1)**(n+1) * n**2
        else:
            output[i] = n*sin( n*arccos(x[i]) ) / sqrt(1-x[i]**2)
    return output

def computeDerivativeMatrix(xlow, xhigh, N):
    Lambda = np.ones((N+1,N+1))
    for i in range(N+1):
        for j in range(N+1):
            j_shift = j+1/2
            Lambda[i,j] = 2/(N+1) * cos(i*j_shift*pi/(N+1))
                
    x = ChebyshevPointsFirstKind(1,-1,N)
    Tp = np.zeros((N+1,N+1))
#     for i in range(N+1):
    for j in range(N+1):
        Tp[:,j] = Tprime(j,x)
    D = 2/(xhigh - xlow) * np.dot(Tp,Lambda)
    return D

def computeLaplacianMatrix(xlow,xhigh,px,ylow,yhigh,py,zlow,zhigh,pz):
    Dx = computeDerivativeMatrix(xlow, xhigh, px)
    Dy = computeDerivativeMatrix(ylow, yhigh, py)
    Dz = computeDerivativeMatrix(zlow, zhigh, pz)
    
    D2x = np.dot(Dx,Dx)
    D2y = np.dot(Dy,Dy)
    D2z = np.dot(Dz,Dz)
    
    laplacian = D2x + D2y + D2z
    
    return laplacian

# def ChebDerivative(xlow, xhigh, N, f,Dopen=None):
def ChebDerivative(f,Dopen):
#     if not Dopen:
#         # if cell hasn't computed Dopen, compute it now.  
#         # generate Lambda
#         Lambda = np.ones((N,N))
#         for i in range(N):
#             for j in range(N):
#                 j_shift = j+1/2
#                 Lambda[i,j] = 2/N * cos(i*j_shift*pi/N)
#                     
#         x = ChebyshevPointsFirstKind(1,-1,N)
#         Tp = np.zeros((N,N))
#     #     for i in range(N+1):
#         for j in range(N):
#             Tp[:,j] = Tprime(j,x)
#         Dopen = 2/(xhigh - xlow) * np.dot(Tp,Lambda)
    return -np.dot(Dopen,f)

# def ChebGradient3D(xlow, xhigh, ylow, yhigh, zlow, zhigh, N, F,DopenX=None):
def ChebGradient3D(DopenX,DopenY,DopenZ,N,F):
 
    DFDX = np.zeros_like(F)
    DFDY = np.zeros_like(F)
    DFDZ = np.zeros_like(F)
    for i in range(N+1):  # assumes Nx=Ny=Nz
        for j in range(N+1):
            DFDX[:,i,j] = -np.dot(DopenX,F[:,i,j]) #ChebDerivative(F[:,i,j],DopenX)
            DFDY[i,:,j] = -np.dot(DopenY,F[i,:,j]) #ChebDerivative(F[i,:,j],DopenY)
            DFDZ[i,j,:] = -np.dot(DopenZ,F[i,j,:]) #ChebDerivative(F[i,j,:],DopenZ)
    return [DFDX,DFDY,DFDZ]

def ChebLaplacian3D(DopenX,DopenY,DopenZ,N,F):
 
    D2FDX2 = np.zeros_like(F)
    D2FDY2 = np.zeros_like(F)
    D2FDZ2 = np.zeros_like(F)
    for i in range(N+1):  # assumes Nx=Ny=Nz
        for j in range(N+1):
            temp = -np.dot(DopenX,F[:,i,j])
            D2FDX2[:,i,j] = -np.dot(DopenX,temp) #ChebDerivative(F[:,i,j],DopenX)
            temp = -np.dot(DopenY,F[i,:,j])
            D2FDY2[i,:,j] = -np.dot(DopenY,temp) #ChebDerivative(F[i,:,j],DopenY)
            temp = -np.dot(DopenZ,F[i,j,:])
            D2FDZ2[i,j,:] = -np.dot(DopenZ,temp) #ChebDerivative(F[i,j,:],DopenZ)
    return D2FDX2 + D2FDY2 + D2FDZ2

def interpolator1Dchebyshev(x,f):
    n = len(x)
    w = np.ones(n)
    for i in range(n):
        w[i] = (-1)**i * np.sin(  (2*i+1)*np.pi / (2*(n-1)+2)  )

    def P(y):
        num = 0
        den = 0
#         print('entering 1D interpolator loop')
        for j in range(len(x)):
            if y==x[j]:
                num += f[j]
                den += 1
            else:
                num += ( w[j]/(y-x[j])*f[j] ) 
                den += ( w[j]/(y-x[j]) )
        
        return num/den

    return np.vectorize(P)

def interpolator2Dchebyshev(x,y,f):
    n = len(x)
    w = np.ones(n)
    for i in range(n):
        w[i] = (-1)**i * np.sin(  (2*i+1)*np.pi / (2*(n-1)+2)  )
    
    def P(xt,yt):  # 2D interpolator.  
        num = 0
        den = 0
        for j in range(n):
            Py = interpolator1Dchebyshev(y, f[j,:])  # calls the 1D interpolator using the y values along xi
            num += ( w[j]/(xt-x[j])*Py(yt) ) 
            den += ( w[j]/(xt-x[j]) )
        
        return num/den

    return np.vectorize(P)

def interpolator2Dchebyshev_oneStep(x,y,f):
    nx = len(x)
    wx = np.ones(nx)
    for i in range(nx):
        wx[i] = (-1)**i * np.sin(  (2*i+1)*np.pi / (2*(nx-1)+2)  )
    
    ny = len(y)
    wy = np.ones(ny)
    for j in range(ny):
        wy[j] = (-1)**j * np.sin(  (2*j+1)*np.pi / (2*(ny-1)+2)  )
    
    def P(xt,yt):  # 2D interpolator.  
        
        num = 0
        for i in range(nx):
            numY = 0
            for j in range(ny):
                numY += ( wy[j]/(yt-y[j])*f[i,j] )
            num +=  ( wx[i]/(xt-x[i]) )*numY
        
        denX=0
        for i in range(nx):
            denX += wx[i]/(xt-x[i])
        
        denY=0
        for j in range(ny):
            denY += wy[j]/(yt-y[j])
        
        den = denX*denY
            
        return num/den

    return np.vectorize(P)

def interpolator3Dchebyshev(x,y,z,f):
    nx = len(x)
    wx = np.ones(nx)
    for i in range(nx):
        wx[i] = (-1)**i * np.sin(  (2*i+1)*np.pi / (2*(nx-1)+2)  )
    
    ny = len(y)
    wy = np.ones(ny)
    for j in range(ny):
        wy[j] = (-1)**j * np.sin(  (2*j+1)*np.pi / (2*(ny-1)+2)  )
        
    nz = len(z)
    wz = np.ones(nz)
    for k in range(nz):
        wz[k] = (-1)**k * np.sin(  (2*k+1)*np.pi / (2*(nz-1)+2)  )
    
    def P3(xt,yt,zt):  # 2D interpolator.  
        
        num = 0
        for i in range(nx):
            numY = 0
            for j in range(ny):
                numZ = 0
                for k in range(nz):
                    numZ += ( wz[k]/(zt-z[k])*f[i,j,k] )
                    
                numY += ( wy[j]/(yt-y[j]) )*numZ
            num +=  ( wx[i]/(xt-x[i]) )*numY
        
        denX=0
        for i in range(nx):
            denX += wx[i]/(xt-x[i])
        
        denY=0
        for j in range(ny):
            denY += wy[j]/(yt-y[j])
            
        denZ=0
        for k in range(nz):
            denZ += wz[k]/(zt-z[k])
        
        den = denX*denY*denZ
            
        return num/den

    return np.vectorize(P3)

def interpolator1Duniform(x,f):
    n = len(x)
    w = np.ones(n)
    for i in range(n):
        w[i] = (-1)**i * factorial(n) / (factorial(i)*factorial(n-i))

    
    def P(y):
        num = 0
        den = 0
        for j in range(len(x)):
            if y==x[j]:
                num += f[j]
                den += 1
            else:
                num += ( w[j]/(y-x[j])*f[j] ) 
                den += ( w[j]/(y-x[j]) )
        
        return num/den

    return np.vectorize(P)


def mkVtkIdList(it):
    vil = vtk.vtkIdList()
    for i in it:
        vil.InsertNextId(int(i))
    return vil
 


def testIntegrationByParts(a,b,c,n):
    
    print('n = ', n)
    # function should have decayed at boundaries a and c
    def f(x):
        return exp(-x**2)
    def fp(x):
        return (-2*x)*exp(-x**2)
    def fpp(x):
        return (4*x**2-2)*exp(-x**2)
    
    def F(x):  # antiderivative of f*f''
        return -1/2 * sqrt(pi/2) * erf( sqrt(2) * x) - x*exp(-2*x**2)
    
    def true_int_F(a,b):
#         return 2*a*exp(-a**2) - 2*b*exp(-b**2)
        return F(b)-F(a)
    
    def g(x):
        return exp(-abs(x))
    def gp(x):
        return -np.sign(x)*exp(-abs(x))
    def gpp(x):
        return exp(-abs(x))
    
    def G(x):  # antiderivative of f*f''
        return g(0) - exp(-abs(x))
    
    def true_int_G(a,b):
        return 1 - 1/2*exp(-abs(2*a)) - 1/2*exp(-abs(2*b))
#         return G(b)-G(a)
    
    # left domain from a to b
    xL = ChebyshevPointsFirstKind(a, b, n)
    wL = weightsFirstKind(a,b,n)
    fL = f(xL)
    gL = g(xL)
    gradL = computeDerivativeMatrix(a,b,n)
    lapL = np.dot(gradL,gradL)
    
    # right domain from b to c
    xR = ChebyshevPointsFirstKind(b, c, n)
    wR = weightsFirstKind(b,c,n)
    fR = f(xR)
    gR = g(xR)
    gradR = computeDerivativeMatrix(b,c,n)
    lapR = np.dot(gradR,gradR)
    
    
    # Analytic value
    I_true = true_int_F(a,c)
    print(I_true)
    
    # Laplacian Method
    I_lap = np.sum( wL * np.dot(lapL,fL)*fL ) + np.sum( wR * np.dot(lapR,fR)*fR )
    print(I_lap)
    
    # Gradient Method
    I_grad = np.sum( -wL * np.dot(gradL,fL)**2 ) + np.sum( -wR * np.dot(gradR,fR)**2 )
    print(I_grad)
    
    lapErrorF = I_lap-I_true
    gradErrorF=I_grad-I_true
    
    
    # Analytic value
    I_true = true_int_G(a,c)
    print(I_true)
    
    # Laplacian Method
    I_lap = np.sum( wL * np.dot(lapL,gL)*gL ) + np.sum( wR * np.dot(lapR,gR)*gR )
    print(I_lap)
    
    # Gradient Method
    I_grad = np.sum( -wL * np.dot(gradL,gL)**2 ) + np.sum( -wR * np.dot(gradR,gR)**2 )
    print(I_grad)
    
    lapErrorG = I_lap-I_true
    gradErrorG=I_grad-I_true
    
    # Numerical correction
    grad_gL = np.dot(-gradL,gL)
    grad_gR = np.dot(-gradR,gR)
    
    # Interpolators for g and gp
    gL_interpolator = interpolator1Dchebyshev(xL,gL)
    gpL_interpolator = interpolator1Dchebyshev(xL,grad_gL)
    
    gR_interpolator = interpolator1Dchebyshev(xR,gR)
    gpR_interpolator = interpolator1Dchebyshev(xR,grad_gR)
    
    
    
    print('grad_gL: ',grad_gL)
    print('grad_gR: ',grad_gR)
    
#     BC_numerically_correctedI_grad = I_grad - ( gL[0]*np.dot(-gradL,gL)[0] - gL[-1]*np.dot(-gradL,gL)[-1]) - (gR[0]*np.dot(-gradR,gR)[0] - gR[-1]*np.dot(-gradR,gR)[-1] )  
    BC_numerically_correctedI_grad = I_grad + ( gL[-1]*grad_gL[-1] - gL[0]*grad_gL[0]) + ( gR[-1]*grad_gR[-1] - gR[0]*grad_gR[0] )  
#     BC_numerically_correctedI_grad = I_grad + ( gL_interpolator(b)*gpL_interpolator(b) - gL_interpolator(a)*gpL_interpolator(a) ) + ( gR_interpolator(c)*gpR_interpolator(c) - gR_interpolator(b)*gpR_interpolator(b) )  
    # Analytic coorection
    BC_analytically_correctedI_grad = I_grad + ( g(b)*gp(-1e-10) - g(a)*gp(a)) + (g(c)*gp(c) - g(b)*gp(1e-10) )  
    
#     BCcorrectedI_grad = I_grad + 2 # for the cusp at 0
#     BCcorrectedI_grad =BCcorrectedI_grad - g(a)*gp(a) - g(b) * gp(b)
    BC_analytically_correctedGradErrorG = BC_analytically_correctedI_grad - I_true
    BC_numerically_correctedGradErrorG = BC_numerically_correctedI_grad - I_true
    
    
#     print('Error in Laplacian: ', I_lap-I_true)
#     print('Error in Gradient: ', I_grad-I_true)
#     print()
    
    return lapErrorF, gradErrorF, lapErrorG, gradErrorG, BC_analytically_correctedGradErrorG, BC_numerically_correctedGradErrorG
    
def sweepIntegrationByParts():
    N = range(2,20)
    laplacianErrorsF = []
    gradientErrorsF = []
    laplacianErrorsG = []
    gradientErrorsG = []
    BC_analytically_correctedG = []
    BC_numerically_correctedG = [] 
    for i in range(len(N)):
        n=N[i]
        lapErrorF, gradErrorF, lapErrorG, gradErrorG, BC_analytical_gradErrorG, BC_numerical_gradErrorG = testIntegrationByParts(-10,0,10,n)
        laplacianErrorsF.append(np.abs(lapErrorF))
        gradientErrorsF.append(np.abs(gradErrorF))
        laplacianErrorsG.append(np.abs(lapErrorG))
        gradientErrorsG.append(np.abs(gradErrorG))
        BC_analytically_correctedG.append(abs( BC_analytical_gradErrorG))
        BC_numerically_correctedG.append(abs( BC_numerical_gradErrorG))
        
        
    plt.figure()
    plt.title(r"$\int f(x)*f''(x) dx$ with and without integration by parts, smooth f")
    plt.loglog(N,laplacianErrorsF,'ro', label='Without IBP')
    plt.loglog(N,gradientErrorsF,'bo', label='With IBP')
    plt.legend()
    
    plt.figure()
    plt.title(r"$\int f(x)\frac{d^2}{dx^2}f(x) dx$; $f=e^{-|x|}$; domain split at the $x=0$ cusp")
    plt.loglog(N,laplacianErrorsG,'ro', label='Without IBP')
    plt.loglog(N,gradientErrorsG,'bo', label='IBP without boundary term')
    plt.loglog(N,BC_analytically_correctedG,'go', label='IBP with analytically corrected cusp')
    plt.loglog(N,BC_numerically_correctedG,'co', label='IBP with numerically corrected cusp')
    plt.legend()
    plt.xlabel("Chebyshev Degree")
    plt.ylabel("Error in Integral")
    plt.show()
    

def testGradientAndLaplacian():
 

    N = range(2,20)
    gradientErrors=[]
    laplacianErrors=[]
    
    gradientErrors7=[]
    laplacianErrors7=[]
    for i in range(len(N)):
        
#         nx = ny = nz = 12
        nx = ny = nz = N[i]
        a=-1
        b=2
        xf = ChebyshevPointsFirstKind(a, b, nx)
        xs = ChebyshevPointsSecondKind(a,b, nx)
        
        wf = weightsFirstKind(a,b,nx)
        ws = weightsSecondKind(a,b,nx)
        
        df = computeDerivativeMatrix(a,b,nx)
    #     y = ChebyshevPointsFirstKind(-1, 0, ny)
    #     z = ChebyshevPointsFirstKind(-1, 0, nz)
    #     print(xf)
    #     print(xs)
    #     print()
    #     print(wf)
    #     print(ws)
        
        
        def square(x):
            return x**2
        def square_int(a,b):
            return 1/3* (b**3 - a**3)
        def square_prime(x):
            return 2*x
        def square_pp(x):
            return 2*np.ones_like(x)
        
        def eight(x):
            return x**8
        def eight_int(a,b):
            return 1/9*(b**9-a**9)
        def eight_prime(x):
            return 8*x**7
        
        
        def seven(x):
            return x**7
        def seven_int(a,b):
            return 1/8*(b**8-a**8)
        def seven_prime(x):
            return 7*x**6
        def seven_pp(x):
            return 42*x**5
        
        def exp(x):
            return np.exp(x)
        def exp_int(a,b):
            return np.exp(b)-np.exp(a)
        def exp_prime(x):
            return exp(x)
        def exp_pp(x):
            return np.exp(x)
        
        def ErrorNorm(x):
            return np.sqrt(np.sum(x**2))
            
#         func_f = square(xf)
#         sum_f = np.sum(func_f*wf)
#         grad_f = np.dot(-df,func_f)
#         lap_f = np.dot(-df,grad_f)
#         
#         func_s = square(xs)
#         sum_s = np.sum(func_s*ws)
#         
#         print("Square function")
#     #     print('int error: ', sum_f-square_int(a,b))
#     #     print(square_prime(xf))
#     #     print(grad_f)
#         print('gradient error: ', ErrorNorm( grad_f - square_prime(xf)) )
#         print('laplacian error: ', ErrorNorm( lap_f - square_pp(xf)) )
#     # #     print('Sum s error: ', sum_s-square_int(a,b))
#     # #     print('Sum simp error: ', 1/3*sum_s+2/3*sum_f-1*square_int(-1,1))
#     #     

        func_f = seven(xf)
        sum_f = np.sum(func_f*wf)
        grad_f = np.dot(-df,func_f)
        Lap = np.dot(df,df)
        lap_f = np.dot(Lap,func_f)
        gradientErrors7.append(ErrorNorm( grad_f - seven_prime(xf)))
        laplacianErrors7.append(ErrorNorm(  lap_f - seven_pp(xf)))
        print(seven_pp(xf))
        print(lap_f)
        print('prime error: ', ErrorNorm( grad_f - seven_prime(xf)) )
        print('lap error: ', ErrorNorm( lap_f - seven_pp(xf)) )
        print('relative lap error: ', ErrorNorm( lap_f - seven_pp(xf)) /ErrorNorm(np.abs(seven_pp(xf) )) )
        print('norm of lap: ', ErrorNorm(seven_pp(xf)) )
    
    #     func_s = seven(xs)
    #     sum_s = np.sum(func_s*ws)
    #     
    #     print("seven power function")
    #     print('Sum f error: ', sum_f-seven_int(a,b))
    # #     print('Sum s error: ', sum_s-seven_int(a,b))
    # #     print(sum_f)
    # #     print(seven_int(a,b))
    # #     print('Sum simp error: ', 1/3*sum_s+2/3*sum_f-1*eight_int(-1,1))
    # 
    #     func_f = eight(xf)
    #     sum_f = np.sum(func_f*wf)
    #     
    #     func_s = eight(xs)
    #     sum_s = np.sum(func_s*ws)
    #     
    #     print("Eight power function")
    #     print('Sum f error: ', sum_f-eight_int(a,b))
    # #     print('Sum s error: ', sum_s-eight_int(a,b))
    # #     print(sum_f)
    # #     print(eight_int(a,b))
    #     
    
        print()
        func_f = exp(xf)
        sum_f = np.sum(func_f*wf)
        grad_f = np.dot(-df,func_f)
        lap_f = np.dot(-df,grad_f)
#         print(exp_prime(xf))
#         print(grad_f)
#         print('gradient error: ', ErrorNorm( grad_f - exp_prime(xf)) )
#         print('laplacian error: ', ErrorNorm( lap_f - exp_pp(xf)) )    
#         print()
        
        gradientErrors.append(ErrorNorm( grad_f - exp_prime(xf)))
        laplacianErrors.append(ErrorNorm(  lap_f - exp_pp(xf)))
    
    plt.figure()
    plt.title(r'$L^2$ Errors in Derivatives of $e^x$ on [%i,%i]' %(a,b))
    plt.loglog(N,gradientErrors,'bo', label='Gradient')
    plt.loglog(N,laplacianErrors,'ro', label='Laplacian')
    plt.legend()
    
    plt.figure()
    plt.title(r'$L^2$ Errors in Derivatives of $x^7$ on [%i,%i]' %(a,b))
    plt.loglog(N,gradientErrors7,'bo', label='Gradient')
    plt.loglog(N,laplacianErrors7,'ro', label='Laplacian')
    plt.legend()
    
    
    plt.show()
    







def testInterpolationBetweenMeshes_OLD():
    
    coarseP = 0
    fineP = 2#coarseP+2
    
    # Test single cell
    tempX_coarse = ChebyshevPointsFirstKind(0,1,coarseP)
    tempY_coarse = ChebyshevPointsFirstKind(0,1,coarseP)
    tempZ_coarse = ChebyshevPointsFirstKind(0,1,coarseP)
    print("Coarse Z = ", tempZ_coarse)

    coarseX=np.zeros((coarseP+1)**3)
    coarseY=np.zeros((coarseP+1)**3)
    coarseZ=np.zeros((coarseP+1)**3)
    
    idx=0
    for i in range(coarseP+1):
        for j in range(coarseP+1):
            for k in range(coarseP+1):
                coarseX[idx] = tempX_coarse[i]
                coarseY[idx] = tempY_coarse[j]
                coarseZ[idx] = tempZ_coarse[k]
                idx+=1
    coarseW = weights3DFirstKind(0,1,coarseP,0,1,coarseP,0,1,coarseP).reshape([(coarseP+1)**3,])
    
    tempX_fine = ChebyshevPointsFirstKind(0,1,fineP)
    tempY_fine = ChebyshevPointsFirstKind(0,1,fineP)
    tempZ_fine = ChebyshevPointsFirstKind(0,1,fineP)
    print("Fine Z = ", tempZ_fine)

    fineX=np.zeros((fineP+1)**3)
    fineY=np.zeros((fineP+1)**3)
    fineZ=np.zeros((fineP+1)**3)
    
    idx=0
    for i in range(fineP+1):
        for j in range(fineP+1):
            for k in range(fineP+1):
                fineX[idx] = tempX_fine[i]
                fineY[idx] = tempY_fine[j]
                fineZ[idx] = tempZ_fine[k]
                idx+=1
    fineW = weights3DFirstKind(0,1,fineP,0,1,fineP,0,1,fineP).reshape([(fineP+1)**3,])
    

#     singularities=[]
#     
#     fineIdx=0
#     for i in range(fineP+1):
#         for j in range(fineP+1):
#             for k in range(fineP+1):
#                 
#                 coarseIdx=0
#                 for ii in range(coarseP+1):
#                     for jj in range(coarseP+1):
#                         for kk in range(coarseP+1):
#                             
#                             if (  (abs(fineX[fineIdx]-coarseX[coarseIdx])<1e-12) and 
#                                   (abs(fineY[fineIdx]-coarseY[coarseIdx])<1e-12) and 
#                                   (abs(fineZ[fineIdx]-coarseZ[coarseIdx])<1e-12) ):
#                                 singularities.append( [coarseIdx,fineIdx])
#                             
#                             coarseIdx+=1
#                             
#                             
#                 fineIdx+=1
                
#     print("singularities: ", singularities)
    
#     x_exact=[]
#     for i in range(fineP+1):
#         for ii in range(coarseP+1):
#             if 
    
    coarseF = coarseX**4 * coarseY**4 
    coarseG =  coarseZ**6
    fineF = fineX**4 * fineY**4 
    fineG =  fineZ**6
    analyticIntegral = 1/175
    coarseComputedIntegral = np.sum(coarseF*coarseG*coarseW)
    fineComputedIntegral = np.sum(fineF*fineG*fineW)

    
    
    interpolatedF = interpolateBetweenTwoMeshes(coarseX, coarseY, coarseZ, coarseF, coarseP,
                                                fineX, fineY, fineZ, fineP)
    

    interpolatedIntegral = np.sum(interpolatedF*fineG*fineW)
    
    print("Expected integral = %f" %analyticIntegral)
    print("Coarse computed integral         = %f, error %1.3e" %(coarseComputedIntegral,coarseComputedIntegral-analyticIntegral) )
    print("Fine computed integral           = %f, error %1.3e" %(fineComputedIntegral,fineComputedIntegral-analyticIntegral) )
    print("Interpolated computed integral   = %f, error %1.3e" %(interpolatedIntegral,interpolatedIntegral-analyticIntegral) )
    
#     print("coarseF=      ",coarseF[int(len(coarseF)/2)])
#     print("fineF=        ",fineF[int(len(fineF)/2)])
#     print("interpolatedF=",interpolatedF)
    
    
#     # Test 8 cells
#     
#     coarseX=np.zeros(8*(coarseP+1)**3)
#     coarseY=np.zeros(8*(coarseP+1)**3)
#     coarseZ=np.zeros(8*(coarseP+1)**3)
#     coarseW=np.array([])
#         
#     idx=0
#     for xlow in [-1,0]:
#         tempX = ChebyshevPointsFirstKind(xlow,xlow+1,coarseP)
#         for ylow in [-1,0]:
#             tempY = ChebyshevPointsFirstKind(xlow,xlow+1,coarseP)
#             for zlow in [-1,0]:
#                 tempZ = ChebyshevPointsFirstKind(xlow,xlow+1,coarseP)
#                 coarseW = np.append(coarseW, weights3DFirstKind(xlow,xlow+1,coarseP,ylow,ylow+1,coarseP,zlow,zlow+1,coarseP).reshape([(coarseP+1)**3,]) )
#                 
#                 
#                 for i in range(coarseP+1):
#                     for j in range(coarseP+1):
#                         for k in range(coarseP+1):
#                             coarseX[idx] = tempX[i]
#                             coarseY[idx] = tempY[j]
#                             coarseZ[idx] = tempZ[k]
#                             idx+=1
#                 
#                 
#                 
#     fineX=np.zeros(8*(fineP+1)**3)
#     fineY=np.zeros(8*(fineP+1)**3)
#     fineZ=np.zeros(8*(fineP+1)**3)
#     fineW=np.array([])
#     
#     idx=0
#     for xlow in [-1,0]:
#         tempX = ChebyshevPointsFirstKind(xlow,xlow+1,fineP)
#         for ylow in [-1,0]:
#             tempY = ChebyshevPointsFirstKind(xlow,xlow+1,fineP)
#             for zlow in [-1,0]:
#                 tempZ = ChebyshevPointsFirstKind(xlow,xlow+1,fineP)
#                 
#                 fineW = np.append(fineW, weights3DFirstKind(xlow,xlow+1,fineP,ylow,ylow+1,fineP,zlow,zlow+1,fineP).reshape([(fineP+1)**3,]) )
#                 
#                 
#                 for i in range(fineP+1):
#                     for j in range(fineP+1):
#                         for k in range(fineP+1):
#                             fineX[idx] = tempX[i]
#                             fineY[idx] = tempY[j]
#                             fineZ[idx] = tempZ[k]
#                             idx+=1
#                 
# 
#     
#     print(np.shape(fineX))
#     print(np.shape(fineW))
#     
#     coarseF = coarseX**4 + coarseY**4 + coarseZ**4
#     fineF = fineX**4 + fineY**4 + fineZ**4
#     analyticIntegral = 8*0.6
#     
#     coarseComputedIntegral = np.sum(coarseF*coarseW)
#     fineComputedIntegral = np.sum(fineF*fineW)
#     print("Expected integral = %f" %analyticIntegral)
#     print("Coarse computed integral         = %f, error %1.3e" %(coarseComputedIntegral,coarseComputedIntegral-analyticIntegral) )
#     print("Fine computed integral           = %f, error %1.3e" %(fineComputedIntegral,fineComputedIntegral-analyticIntegral) )
#     
#     
#     interpolatedF = interpolateBetweenTwoMeshes(coarseX, coarseY, coarseZ, coarseF, coarseP,
#                                                 fineX, fineY, fineZ, fineP)
# 
#     interpolatedIntegral = np.sum(interpolatedF*fineW)
#      
#     print("Interpolated computed integral   = %f, error %1.3e" %(interpolatedIntegral,interpolatedIntegral-analyticIntegral) )
#     
    
    return


def testInterpolationBetweenMeshes():
    
    
    order=6

    
    # Test single cell
    tempX_coarse = ChebyshevPointsFirstKind(0,1,order)
    tempY_coarse = ChebyshevPointsFirstKind(0,1,order)
    tempZ_coarse = ChebyshevPointsFirstKind(0,1,order)
    print("Coarse Z = ", tempZ_coarse)

    coarseX=np.zeros((order+1)**3)
    coarseY=np.zeros((order+1)**3)
    coarseZ=np.zeros((order+1)**3)
    
    idx=0
    for i in range(order+1):
        for j in range(order+1):
            for k in range(order+1):
                coarseX[idx] = tempX_coarse[i]
                coarseY[idx] = tempY_coarse[j]
                coarseZ[idx] = tempZ_coarse[k]
                idx+=1
    coarseW = weights3DFirstKind(0,1,order,0,1,order,0,1,order).reshape([(order+1)**3,])
    
    
    
    fineX=np.zeros(8*(order+1)**3)
    fineY=np.zeros(8*(order+1)**3)
    fineZ=np.zeros(8*(order+1)**3)
    fineW=np.empty(0)
    idx=0
    
    
    for i in range(2):
        for j in range(2):
            for k in range(2):
    
                xlow = 0+0.5*i
                xhigh = 0.5+0.5*i
                
                ylow = 0+0.5*j
                yhigh = 0.5+0.5*j
                
                zlow = 0+0.5*k
                zhigh = 0.5+0.5*k
                
                
                tempX_fine = ChebyshevPointsFirstKind(0+0.5*i,0.5+0.5*i,order)
                tempY_fine = ChebyshevPointsFirstKind(0+0.5*j,0.5+0.5*j,order)
                tempZ_fine = ChebyshevPointsFirstKind(0+0.5*k,0.5+0.5*k,order)
#                 print("tempZ_fine = ", tempZ_fine)

    
                fineW = np.append(fineW, weights3DFirstKind(xlow,xhigh,order,ylow,yhigh,order,zlow,zhigh,order).reshape([(order+1)**3,]) )
    
                for ii in range(order+1):
                    for jj in range(order+1):
                        for kk in range(order+1):
                            fineX[idx] = tempX_fine[ii]
                            fineY[idx] = tempY_fine[jj]
                            fineZ[idx] = tempZ_fine[kk]
                            idx+=1
#     fineW = weights3DFirstKind(0,1,fineP,0,1,fineP,0,1,fineP).reshape([(fineP+1)**3,])
    
    
    coarseF = coarseX**4 * coarseY**4 
    coarseG =  coarseZ**6
    fineF = fineX**4 * fineY**4 
    fineG =  fineZ**6
    analyticIntegral = 1/175
    coarseComputedIntegral = np.sum(coarseF*coarseG*coarseW)
    fineComputedIntegral = np.sum(fineF*fineG*fineW)

    pointsPerCoarseCell=np.array([(order+1)**3])
    pointsPerFineCell=np.array([8*(order+1)**3])
    
    
    start=time.time()
    for i in range(10):
        interpolatedF = interpolateBetweenTwoMeshes(coarseX, coarseY, coarseZ, coarseF, pointsPerCoarseCell,
                                                    fineX, fineY, fineZ, pointsPerFineCell)
    end=time.time()
    print("Original time: ", end-start)
    
    numberOfCells=1
    interpolatedF_extern = interpolation_wrapper.callInterpolator(coarseX, coarseY, coarseZ, coarseF, pointsPerCoarseCell,
                                                fineX, fineY, fineZ, pointsPerFineCell, 
                                                numberOfCells, order)
    start=time.time()
    for i in range(10):
        interpolatedF_extern = interpolation_wrapper.callInterpolator(coarseX, coarseY, coarseZ, coarseF, pointsPerCoarseCell,
                                                fineX, fineY, fineZ, pointsPerFineCell, 
                                                numberOfCells, order)
    end=time.time()
    print("External time: ", end-start)
    
    print("Max error in external interpolation: ", np.max( np.abs( interpolatedF_extern-interpolatedF ) ) )
    

    interpolatedIntegral = np.sum(interpolatedF*fineG*fineW)
    external_interpolatedIntegral = np.sum(interpolatedF_extern*fineG*fineW)
    
    print("Expected integral = %f" %analyticIntegral)
    print("Coarse computed integral         = %f, error %1.3e" %(coarseComputedIntegral,coarseComputedIntegral-analyticIntegral) )
    print("Fine computed integral           = %f, error %1.3e" %(fineComputedIntegral,fineComputedIntegral-analyticIntegral) )
    print("Interpolated computed integral   = %f, error %1.3e" %(interpolatedIntegral,interpolatedIntegral-analyticIntegral) )
    print("External Interpolated  integral  = %f, error %1.3e" %(external_interpolatedIntegral,external_interpolatedIntegral-analyticIntegral) )
    
    
    
    return

if __name__=="__main__":    
    testInterpolationBetweenMeshes()
    
    
    
    
    
    
    
    
    
    
#     testIntegrationByParts(-5,0,5,5)
#     testIntegrationByParts(-5,0,5,10)
#     testIntegrationByParts(-5,0,5,15)
#     testIntegrationByParts(-5,0,5,25)
#     testIntegrationByParts(-5,0,5,50)
#    
#     sweepIntegrationByParts()
#     testGradientAndLaplacian()

#     
#     func_s = exp(xs)
#     grad_s = np.dot(-df,func_s)
#     print(exp_prime(xs))
#     print(grad_s)
#     print('prime error: ', ErrorNorm( grad_s - exp_prime(xs)) )
#     sum_s = np.sum(func_s*ws)
#     
#     print("exp function")
#     print('Sum f error: ', sum_f-exp_int(a,b))
# #     print('Sum s error: ', sum_s-exp_int(a,b))
# #     print('Sum simp error: ', 1/3*sum_s+2/3*sum_f-1*exp_int(-1,1))
# 
# #     print(square_int(-1,1))
    


#     nx = ny = nz = 8
#     x = ChebyshevPointsFirstKind(-1, 0, nx)
#     y = ChebyshevPointsFirstKind(-1, 0, ny)
#     z = ChebyshevPointsFirstKind(-1, 0, nz)
# #     print(x)
# #     x = mapToMinusOneToOne(-2,1,x)
# #     print(x)
# #     print(y)
# #     print(z)
#     f = np.zeros((nx,ny,nz))
#      
#     for i in range(nx):
#         for j in range(ny):
#             for k in range(nz):
#                 f[i,j,k] = x[i]**4 * exp(y[j]**2-y[j]) * z[k]**1
# #                 f[i,j,k] = exp(-sqrt( abs(x[i]**2 +y[j]**2+ z[k]**2)) )
# #                 f[i,j,k] = x[i]**2 * y[j]**1 * z[k]**1 + 6
# #                 f[i,j,k] = (4*x[i]**3-3*x[i]) * y[j]**1 * z[k]**1
# #                 f[i,j,k] = x[i]**2 * y[j]**1 * z[k]**1
#                  
# #     coefficients = computeCoefficicents(x,y,z,f)
#     coefficients = computeCoefficicents(f)
#      
#  
#     # reconstruct f from coefficients
#  
#     g = np.zeros_like(f)
#     x = ChebyshevPointsFirstKind(-1, 1, nx)
#     y = ChebyshevPointsFirstKind(-1, 1, ny)
#     z = ChebyshevPointsFirstKind(-1, 1, nz)
#     for i in range(nx):
#         for j in range(ny):
#             for k in range(nz):
#                  
#                 for ell in range(nx):
#                     for m in range(ny):
#                         for n in range(nz):
#                             g[i,j,k] += coefficients[ell,m,n] * cos(ell*arccos(x[i])) * cos(m*arccos(y[j])) * cos(n*arccos(z[k]))
# #                             g[i,j,k] += coefficients[ell,m,n] * cos(ell*x[i]) * cos(m*y[j]) * cos(n*z[k])
# #                             g[i,j,k] += coefficients[ell,m,n] * cos(x[i])**ell * cos(y[j])**m * cos(z[k])**n
#      
#      
# #     print('\nf = ', f,'\n')
# #     print('g = ', g,'\n')
# # #     
# #     print('f-g = ',f-g)
#      
#     print('\nMax error: ', np.max(f-g))
#      
#      
#     plt.figure()
#     for n in range(nx+ny+nz):
#         val = sumChebyshevCoefficicentsGreaterThanOrderQ(f,n)
#         print(val)
#         plt.semilogy(n,val,'ko')
#     plt.title('Absolute Sum of Coefficients Greater Than Specified Order')
#     plt.xlabel('i+j+k > ')
#     plt.ylabel('Sum')
# #     plt.show()
#      
#     plt.figure()
# #     if coefficients[0,0,0]==0.0:
# #         coefficients[0,0,0] = 1e-17*np.random.rand()
# #     plt.title('Chebyshev Coefficients for $f = x^4 * e^{y^2-y} * z$')
#     plt.title('Chebyshev Coefficients for $f = e^{-r}$')
#     plt.semilogy(0,abs(coefficients[0,0,0]),'bo',label='x order')
#     plt.semilogy(0,abs(coefficients[0,0,0]),'g^',label='y order')
#     plt.semilogy(0,abs(coefficients[0,0,0]),'rx',label='z order')
#     for i in range(nx):
#         for j in range(ny):
#             for k in range(nz):
# #                 if coefficients[i,j,k]==0.0:
# #                     coefficients[i,j,k] = 1e-17*np.random.rand()
#                 plt.semilogy(i,np.abs(coefficients[i,j,k]),'bo')
#                 plt.semilogy(j,np.abs(coefficients[i,j,k]),'g^')
#                 plt.semilogy(k,np.abs(coefficients[i,j,k]),'rx')
#    
#     plt.legend()
#        
#        
#     plt.figure()
# #     if coefficients[0,0,0]==0.0:
# #         coefficients[0,0,0] = 1e-17*np.random.rand()
#     plt.title('Largest Chebyshev Coefficients for $f = x^4 * e^{y^2-y} * z$')
# #     plt.title('Largest Chebyshev Coefficients for $f = e^{-r}$')
#     plt.semilogy(0,np.max(abs(coefficients[0,:,:])),'bo',label='max x')
#     plt.semilogy(0,np.max(abs(coefficients[:,0,:])),'g^',label='max y')
#     plt.semilogy(0,np.max(abs(coefficients[:,:,0])),'rx',label='max z')
#     for i in range(nx):
# #         for j in range(ny):
# #             for k in range(nz):
# # #                 if coefficients[i,j,k]==0.0:
# # #                     coefficients[i,j,k] = 1e-17*np.random.rand()
#         plt.semilogy(i,np.max(np.abs(coefficients[i,:,:])),'bo')
#         plt.semilogy(i,np.max(np.abs(coefficients[:,i,:])),'g^')
#         plt.semilogy(i,np.max(np.abs(coefficients[:,:,i])),'rx')
#    
#     plt.legend()
#       
#       
#     plt.show()
    
    
    
    
    
    
    
#     ## REFINEMENT TEST
#     nx = ny = nz = 5
#     plt.figure()
#     plt.title('Absolute Sum of Coefficients Greater Than Specified Order')
#     plt.xlabel('Order')
#     plt.ylabel('Sum')
#      
#     for ii in range(8):
#         print('ii = ', ii)
#         x = ChebyshevPointsFirstKind(0, 1/(2**ii), nx)
#         y = ChebyshevPointsFirstKind(0, 1/(2**ii), ny)
#         z = ChebyshevPointsFirstKind(0, 1/(2**ii), nz)
#         f = np.zeros((nx,ny,nz))
#          
#         for i in range(nx):
#             for j in range(ny):
#                 for k in range(nz):
#                     f[i,j,k] = exp(-sqrt( abs(x[i]**2 +y[j]**2+ z[k]**2)) )
#          
#          
#          
#          
#         if ii==0:
#             plt.semilogy(0,sumChebyshevCoefficicentsGreaterThanOrderQ(f,0),'ko',label='Refinement Level %i' %ii)
#         elif ii==1:
#             plt.semilogy(0,sumChebyshevCoefficicentsGreaterThanOrderQ(f,0),'bo',label='Refinement Level %i' %ii)
#         elif ii==2:
#             plt.semilogy(0,sumChebyshevCoefficicentsGreaterThanOrderQ(f,0),'go',label='Refinement Level %i' %ii)
#         elif ii==3:
#             plt.semilogy(0,sumChebyshevCoefficicentsGreaterThanOrderQ(f,0),'ro',label='Refinement Level %i' %ii)
#         elif ii==4:
#             plt.semilogy(0,sumChebyshevCoefficicentsGreaterThanOrderQ(f,0),'mo',label='Refinement Level %i' %ii)
#         elif ii==5:
#             plt.semilogy(0,sumChebyshevCoefficicentsGreaterThanOrderQ(f,0),'k^',label='Refinement Level %i' %ii)
#         elif ii==6:
#             plt.semilogy(0,sumChebyshevCoefficicentsGreaterThanOrderQ(f,0),'b^',label='Refinement Level %i' %ii)
#         elif ii==7:
#             plt.semilogy(0,sumChebyshevCoefficicentsGreaterThanOrderQ(f,0),'g^',label='Refinement Level %i' %ii)
#         elif ii==8:
#             plt.semilogy(0,sumChebyshevCoefficicentsGreaterThanOrderQ(f,0),'r^',label='Refinement Level %i' %ii)
#         elif ii==9:
#             plt.semilogy(0,sumChebyshevCoefficicentsGreaterThanOrderQ(f,0),'m^',label='Refinement Level %i' %ii)
#                  
#         for n in range(1,nx+ny+nz):
#             if ii==0:
#                 plt.semilogy(n,sumChebyshevCoefficicentsGreaterThanOrderQ(f,n),'ko')
#             elif ii==1:
#                 plt.semilogy(n,sumChebyshevCoefficicentsGreaterThanOrderQ(f,n),'bo')
#             elif ii==2:
#                 plt.semilogy(n,sumChebyshevCoefficicentsGreaterThanOrderQ(f,n),'go')
#             elif ii==3:
#                 plt.semilogy(n,sumChebyshevCoefficicentsGreaterThanOrderQ(f,n),'ro')
#             elif ii==4:
#                 plt.semilogy(n,sumChebyshevCoefficicentsGreaterThanOrderQ(f,n),'mo')
#             elif ii==5:
#                 plt.semilogy(n,sumChebyshevCoefficicentsGreaterThanOrderQ(f,n),'k^')
#             elif ii==6:
#                 plt.semilogy(n,sumChebyshevCoefficicentsGreaterThanOrderQ(f,n),'b^')
#             elif ii==7:
#                 plt.semilogy(n,sumChebyshevCoefficicentsGreaterThanOrderQ(f,n),'g^')
#             elif ii==8:
#                 plt.semilogy(n,sumChebyshevCoefficicentsGreaterThanOrderQ(f,n),'r^')
#             elif ii==9:
#                 plt.semilogy(n,sumChebyshevCoefficicentsGreaterThanOrderQ(f,n),'m^')
#      
#     plt.legend()
#     plt.show()
         
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
# #     import matplotlib.pyplot as plt
# #     
# #     r = np.linspace(0.1,10,500)
# #     print(r)
# #     N = 8
# #     eta = np.sqrt(2*0.3)
# #     A = 5
# #     k=5
# # #     for k in range(3,6):
# # #         D = phaniMeshDensity(A,N,eta,k,r)
# # #     
# # #         plt.plot(r,D,label='k=%i'%k)
# #         
# #     D = meshDensity(r,500,'LW1')
# #     plt.plot(r,D,label='LW1')
# #     D = meshDensity(r,500,'LW2')
# #     plt.plot(r,D,label='LW2')
# #     D = meshDensity(r,1000,'LW3')
# #     plt.plot(r,D,label='LW3')
# #     D = meshDensity(r,1000,'LW4')
# #     plt.plot(r,D,label='LW4')
# #     D = meshDensity(r,1000,'LW5')
# #     plt.plot(r,D,label='LW5')
# #     plt.legend()
# #     plt.show()
# 
# #     x = ChebyshevPointsFirstKind(-1, 1, 4)
# #     print(x)
#     Nx = Ny = Nz = 5
#     xlow = -5
#     xhigh = -1
#     ylow=0
#     yhigh=2
#     zlow=0
#     zhigh=1
#     xw = weights(xlow, xhigh, Nx)
#     yw = weights(ylow, yhigh, Ny)
#     zw = weights(zlow, zhigh, Nz)
#     
# #     print('xw: ', xw)
# #     print('yw: ', yw)
# #     print('zw: ', zw)
# 
#     
#     W = weights3D(xlow,xhigh,Nx,ylow,yhigh,Ny,zlow,zhigh,Nz)
# #     print(W[0,0,0]) # should return xw[0]*yx[1]*zw[1]
# #     print(W[1,0,0]) # should return xw[0]*yx[1]*zw[1]
# #     print(W[2,0,0]) # should return xw[0]*yx[1]*zw[1]
# #     print(W)
#     
#     for i in range(Nx):
#         for j in range(Ny):
#             for k in range(Nz):
#                 err = W[i,j,k] - xw[i]*yw[j]*zw[k]
#                 if err > 0.0: print(err)
#     print('Passed')

    
    