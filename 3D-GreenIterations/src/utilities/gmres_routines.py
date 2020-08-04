import numpy as np
import scipy as sp
from scipy.sparse.linalg import gmres
from scipy.sparse.linalg import LinearOperator
import time

import scipy.sparse.linalg as la

import BaryTreeInterface as BT


class gmres_counter(object):
    def __init__(self, disp=True):
        self._disp = disp
        self.niter = 0
    def __call__(self, rk=None):
        self.niter += 1
        if self._disp:
            print('iter %3i\trk = %s' % (self.niter, str(rk)))


def matrix_free(x):
    b = np.zeros_like(x)
    b[0] = 1*x[0] + 2*x[1]
    b[1] = 2*x[0] + 3*x[1]
    return b


def direct_sum_closure(x,y,z):
    def direct_sum(psi):
#         print("calling direct_sum")
        phi = np.zeros_like(psi)
        
        for i in range(len(psi)):
            for j in range(len(psi)):
                if j!=i:
                    dx = x[i]-x[j]
                    dy = y[i]-y[j]
                    dz = z[i]-z[j]
                    r = np.sqrt(dx*dx + dy*dy + dz*dz)
                    
                    phi[i] += psi[j]/r
        return phi
    return direct_sum



def treecode_closure(Nt, Ns,
                     Xt, Yt, Zt,
                     Xs, Ys, Zs, Ws,
                     kernel, numberOfKernelParameters, kernelParameters,
                     singularity, approximation, computeType,
                     GPUpresent, treecode_verbosity, 
                     theta, treecodeDegree, maxPerSourceLeaf, maxPerTargetLeaf):
    
    def treecode(RHO):
        
#         output = BT.callTreedriver(  Nt, Ns,
#                              Xt, Yt, Zt, np.copy(RHO),
#                              Xs, Ys, Zs, np.copy(RHO), Ws,
#                              kernel, numberOfKernelParameters, kernelParameters,
#                              singularity, approximation, computeType,
#                              treecodeDegree, theta, maxPerSourceLeaf, maxPerTargetLeaf,
#                             GPUpresent, verbosity
#                             )
#         print("calling treecode in GMRES...")
        output = BT.callTreedriver(  Nt, Ns,
                                 np.copy(Xt), np.copy(Yt), np.copy(Zt), np.copy(RHO),
                                 np.copy(Xs), np.copy(Ys), np.copy(Zs), np.copy(RHO), np.copy(Ws),
                                 kernel, numberOfKernelParameters, kernelParameters,
                                 singularity, approximation, computeType,
                                 GPUpresent, treecode_verbosity, 
                                 theta=theta, degree=treecodeDegree, sourceLeafSize=maxPerSourceLeaf, targetLeafSize=maxPerTargetLeaf, sizeCheck=1.0)
                           
        return output
    
    return treecode



if __name__=="__main__":
    print("running gmres_routines.py")
    
#     N = 30
#     X = np.random.rand(N)
#     Y = np.random.rand(N)
#     Z = np.random.rand(N)
#     
#     RHO  = np.ones(N)
#     
#     b = np.ones(N)
#     
#     DS=direct_sum_closure(x,y,z)
#     A = LinearOperator( (N,N), matvec=DS)
#     x2, exitCode = gmres(A,b)
#     print("Result: ",x2)
#     
#     for i in range(N):
#         print(x[i],y[i],z[i])
#         
        
    # set treecode parameters
    N=120
    maxPerSourceLeaf = 5
    maxPerTargetLeaf = 5
    GPUpresent = False
    theta = 0.8
    treecodeDegree = 3
    gaussianAlpha = 1.0
    verbosity = 0
     
    approximation = BT.Approximation.LAGRANGE
    singularity   = BT.Singularity.SUBTRACTION
    computeType   = BT.ComputeType.PARTICLE_CLUSTER
     
    kernel = BT.Kernel.COULOMB
    numberOfKernelParameters = 1
    kernelParameters = np.array([1.0])
 
 
    # initialize some random data
    np.random.seed(1)
    RHO = np.random.rand(N)
    X = np.random.rand(N)
    Y = np.random.rand(N)
    Z = np.random.rand(N)
    W = np.ones(N)   # W stores quadrature weights for convolution integrals.  For particle simulations, set = ones.
    
    DS=direct_sum_closure(X,Y,Z)
    
    
    TC=treecode_closure( N, N,
                     np.copy(X), np.copy(Y), np.copy(Z),
                     X, Y, Z, W,
                     kernel, numberOfKernelParameters, kernelParameters,
                     singularity, approximation, computeType,
                     GPUpresent, verbosity, 
                     theta, treecodeDegree, maxPerSourceLeaf, maxPerTargetLeaf)
    
    
    
    b = np.random.rand(N)
    
#     D = LinearOperator( (N,N), matvec=DS)
#     counterDS = gmres_counter()
#     xDS, exitCode = gmres(D,b,callback=counterDS)
#     print("DS Result: ",xDS)
    T = LinearOperator( (N,N), matvec=TC)
    counterTC = gmres_counter(disp=False)
    counterTC2 = gmres_counter(disp=False)
    xTC1, exitCode = la.lgmres(T,b,callback=counterTC,tol=1e-5)
    
    
    
    gmresStart=time.time()
#     xTC1, exitCode = la.gmres(T,b,callback=counterTC,tol=1e-6, maxiter=5000)
    xTC1, exitCode = la.lgmres(T,b,callback=counterTC,tol=1e-5)
    gmresStop=time.time()
    print(" GMRES took %f seconds and %i iterations." %(gmresStop-gmresStart, counterTC.niter) )
    lgmresStart=time.time()
#     xTC2, exitCode = la.lgmres(T,b,callback=counterTC2,tol=1e-5)
    xTC2, exitCode = la.minres(T,b,callback=counterTC2,tol=1e-5)
    lgmresStop=time.time()
#     print("TC Result: ",xTC)
#     print("Difference: ", xTC1-xTC2)
    normDiff = np.sqrt( np.sum( xTC1-xTC2)**2 )
    print("LGMRES took %f seconds and %i iterations." %(lgmresStop-lgmresStart, counterTC2.niter) )
    print("L2 norm of difference: ", normDiff)
#     print(counterTC2)
    

    
    
#     A = np.array([[1,2],[2,3]])
#     print(A)
#     b  = [1,-1]
#     x0 = [1,1]
#     
#     x1, exitCode = gmres(A,b,x0)
#     print(x1)
#     
#     A = LinearOperator( (2,2), matvec=matrix_free)
#     x2, exitCode = gmres(A,b,x0)
#     print(x2)
