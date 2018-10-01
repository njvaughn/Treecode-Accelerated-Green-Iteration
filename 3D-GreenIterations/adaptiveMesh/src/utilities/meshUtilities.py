'''
Mesh utilities for the adaptive mesh refinement.

@author: nathanvaughn
'''
from numpy import pi, cos, arccos, sin, sqrt, exp
import numpy as np
from scipy.special import factorial
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
    
    elif divideCriterion == 'LW3_modified':
        # for order = 3 
        return divideParameter/648.82*(exp(-5*r)* (52 - 102/r + 363/r**2 + 1416/r**3 + 4164/r**4 + 5184/r**5 + 2592/r**6) )**(3/9)
    


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
    u = np.cos(theta)
    x = xlow + (xhigh-xlow)/2*(u+1)
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
        pass