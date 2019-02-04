'''
Mesh utilities for the adaptive mesh refinement.

@author: nathanvaughn
'''
from numpy import pi, cos, arccos, sin, sqrt, exp
import numpy as np
from scipy.special import factorial, comb
import vtk

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
    x = ChebyshevPoints(-1, 1, px)
    y = ChebyshevPoints(-1, 1, py)
    z = ChebyshevPoints(-1, 1, pz)
    
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
    x = ChebyshevPoints(-1, 1, px)
    y = ChebyshevPoints(-1, 1, py)
    z = ChebyshevPoints(-1, 1, pz)
    
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
    x = ChebyshevPoints(-1, 1, px)
    y = ChebyshevPoints(-1, 1, py)
    z = ChebyshevPoints(-1, 1, pz)
    
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
    x = ChebyshevPoints(-1, 1, px)
    y = ChebyshevPoints(-1, 1, py)
    z = ChebyshevPoints(-1, 1, pz)
    
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
    x = ChebyshevPoints(-1, 1, px)
    y = ChebyshevPoints(-1, 1, py)
    z = ChebyshevPoints(-1, 1, pz)
    
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
    
                
                
            

def unscaledWeights(N):
    # generate Lambda
    Lambda = np.ones((N,N))
    for i in range(N):
        for j in range(N):
            j_shift = j+1/2
            Lambda[i,j] = 2/N * cos(i*j_shift*pi/N)

    W = np.zeros(N)
    for i in range(N):
        if i == 0:
            W[i] = 1
        elif i%2==0:
            W[i] = 2/(1-i**2)
        else:
            W[i] = 0
            
    w = np.dot(np.transpose(Lambda),W)
    return w

def weights(xlow, xhigh, N, w=None):
#     if w != None:
    try: 
        return (xhigh - xlow)/2 * w
    except TypeError:
#         print('meshUtilities: Generating weights from scratch')
        return (xhigh - xlow)/2 *unscaledWeights(N)
    
def weights3D(xlow,xhigh,Nx,ylow,yhigh,Ny,zlow,zhigh,Nz,w=None):
    xw = weights(xlow, xhigh, Nx, w)
    yw = weights(ylow, yhigh, Ny, w)
    zw = weights(zlow, zhigh, Nz, w)
    
    return np.outer( np.outer(xw,yw), zw ).reshape([Nx,Ny,Nz])
        
def ChebyshevPoints(xlow, xhigh, N):
    '''
    Generates "open" Chebyshev points. N midpoints in theta.
    '''
    endpoints = np.linspace(np.pi,0,N+1)
    theta = (endpoints[1:] + endpoints[:-1])/2
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
            output[i] = n**2
        elif x[i] == -1:
            output[i] = (-1)**(n+1) * n**2
        else:
            output[i] = n*sin( n*arccos(x[i]) ) / sqrt(1-x[i]**2)
    return output

def computeDerivativeMatrix(xlow, xhigh, N):
    Lambda = np.ones((N,N))
    for i in range(N):
        for j in range(N):
            j_shift = j+1/2
            Lambda[i,j] = 2/N * cos(i*j_shift*pi/N)
                
    x = ChebyshevPoints(1,-1,N)
    Tp = np.zeros((N,N))
#     for i in range(N+1):
    for j in range(N):
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
#         x = ChebyshevPoints(1,-1,N)
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
    for i in range(N):  # assumes Nx=Ny=Nz
        for j in range(N):
            DFDX[:,i,j] = -np.dot(DopenX,F[:,i,j]) #ChebDerivative(F[:,i,j],DopenX)
            DFDY[i,:,j] = -np.dot(DopenY,F[i,:,j]) #ChebDerivative(F[i,:,j],DopenY)
            DFDZ[i,j,:] = -np.dot(DopenZ,F[i,j,:]) #ChebDerivative(F[i,j,:],DopenZ)
    return [DFDX,DFDY,DFDZ]

def ChebLaplacian3D(DopenX,DopenY,DopenZ,N,F):
 
    D2FDX2 = np.zeros_like(F)
    D2FDY2 = np.zeros_like(F)
    D2FDZ2 = np.zeros_like(F)
    for i in range(N):  # assumes Nx=Ny=Nz
        for j in range(N):
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
 

 
if __name__=="__main__":
    import matplotlib.pyplot as plt
    
    nx = ny = nz = 8
    x = ChebyshevPoints(-1, 0, nx)
    y = ChebyshevPoints(-1, 0, ny)
    z = ChebyshevPoints(-1, 0, nz)
#     print(x)
#     x = mapToMinusOneToOne(-2,1,x)
#     print(x)
#     print(y)
#     print(z)
    f = np.zeros((nx,ny,nz))
     
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                f[i,j,k] = x[i]**4 * exp(y[j]**2-y[j]) * z[k]**1
#                 f[i,j,k] = exp(-sqrt( abs(x[i]**2 +y[j]**2+ z[k]**2)) )
#                 f[i,j,k] = x[i]**2 * y[j]**1 * z[k]**1 + 6
#                 f[i,j,k] = (4*x[i]**3-3*x[i]) * y[j]**1 * z[k]**1
#                 f[i,j,k] = x[i]**2 * y[j]**1 * z[k]**1
                 
#     coefficients = computeCoefficicents(x,y,z,f)
    coefficients = computeCoefficicents(f)
     
 
    # reconstruct f from coefficients
 
    g = np.zeros_like(f)
    x = ChebyshevPoints(-1, 1, nx)
    y = ChebyshevPoints(-1, 1, ny)
    z = ChebyshevPoints(-1, 1, nz)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                 
                for ell in range(nx):
                    for m in range(ny):
                        for n in range(nz):
                            g[i,j,k] += coefficients[ell,m,n] * cos(ell*arccos(x[i])) * cos(m*arccos(y[j])) * cos(n*arccos(z[k]))
#                             g[i,j,k] += coefficients[ell,m,n] * cos(ell*x[i]) * cos(m*y[j]) * cos(n*z[k])
#                             g[i,j,k] += coefficients[ell,m,n] * cos(x[i])**ell * cos(y[j])**m * cos(z[k])**n
     
     
#     print('\nf = ', f,'\n')
#     print('g = ', g,'\n')
# #     
#     print('f-g = ',f-g)
     
    print('\nMax error: ', np.max(f-g))
     
     
    plt.figure()
    for n in range(nx+ny+nz):
        val = sumChebyshevCoefficicentsGreaterThanOrderQ(f,n)
        print(val)
        plt.semilogy(n,val,'ko')
    plt.title('Absolute Sum of Coefficients Greater Than Specified Order')
    plt.xlabel('i+j+k > ')
    plt.ylabel('Sum')
#     plt.show()
     
    plt.figure()
#     if coefficients[0,0,0]==0.0:
#         coefficients[0,0,0] = 1e-17*np.random.rand()
#     plt.title('Chebyshev Coefficients for $f = x^4 * e^{y^2-y} * z$')
    plt.title('Chebyshev Coefficients for $f = e^{-r}$')
    plt.semilogy(0,abs(coefficients[0,0,0]),'bo',label='x order')
    plt.semilogy(0,abs(coefficients[0,0,0]),'g^',label='y order')
    plt.semilogy(0,abs(coefficients[0,0,0]),'rx',label='z order')
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
#                 if coefficients[i,j,k]==0.0:
#                     coefficients[i,j,k] = 1e-17*np.random.rand()
                plt.semilogy(i,np.abs(coefficients[i,j,k]),'bo')
                plt.semilogy(j,np.abs(coefficients[i,j,k]),'g^')
                plt.semilogy(k,np.abs(coefficients[i,j,k]),'rx')
   
    plt.legend()
       
       
    plt.figure()
#     if coefficients[0,0,0]==0.0:
#         coefficients[0,0,0] = 1e-17*np.random.rand()
    plt.title('Largest Chebyshev Coefficients for $f = x^4 * e^{y^2-y} * z$')
#     plt.title('Largest Chebyshev Coefficients for $f = e^{-r}$')
    plt.semilogy(0,np.max(abs(coefficients[0,:,:])),'bo',label='max x')
    plt.semilogy(0,np.max(abs(coefficients[:,0,:])),'g^',label='max y')
    plt.semilogy(0,np.max(abs(coefficients[:,:,0])),'rx',label='max z')
    for i in range(nx):
#         for j in range(ny):
#             for k in range(nz):
# #                 if coefficients[i,j,k]==0.0:
# #                     coefficients[i,j,k] = 1e-17*np.random.rand()
        plt.semilogy(i,np.max(np.abs(coefficients[i,:,:])),'bo')
        plt.semilogy(i,np.max(np.abs(coefficients[:,i,:])),'g^')
        plt.semilogy(i,np.max(np.abs(coefficients[:,:,i])),'rx')
   
    plt.legend()
      
      
    plt.show()
    
    
    
    
    
    
    
#     ## REFINEMENT TEST
#     nx = ny = nz = 5
#     plt.figure()
#     plt.title('Absolute Sum of Coefficients Greater Than Specified Order')
#     plt.xlabel('Order')
#     plt.ylabel('Sum')
#      
#     for ii in range(8):
#         print('ii = ', ii)
#         x = ChebyshevPoints(0, 1/(2**ii), nx)
#         y = ChebyshevPoints(0, 1/(2**ii), ny)
#         z = ChebyshevPoints(0, 1/(2**ii), nz)
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
# #     x = ChebyshevPoints(-1, 1, 4)
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

    
    