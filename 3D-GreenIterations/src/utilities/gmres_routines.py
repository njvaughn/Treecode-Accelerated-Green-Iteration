import numpy as np
import scipy as sp
from scipy.sparse.linalg import gmres
from scipy.sparse.linalg import LinearOperator

import BaryTreeInterface as BT


def matrix_free(x):
    b = np.zeros_like(x)
    b[0] = 1*x[0] + 2*x[1]
    b[1] = 2*x[0] + 3*x[1]
    return b


def direct_sum_closure(x,y,z):
    def direct_sum(psi):
        print("calling direct_sum")
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
    N=50
    maxPerSourceLeaf = 5
    maxPerTargetLeaf = 5
    GPUpresent = False
    theta = 0.8
    treecodeDegree = 3
    gaussianAlpha = 1.0
    verbosity = 0
     
    approximation = BT.Approximation.LAGRANGE
    singularity   = BT.Singularity.SKIPPING
    computeType   = BT.ComputeType.PARTICLE_CLUSTER
     
    kernel = BT.Kernel.COULOMB
    numberOfKernelParameters = 0
    kernelParameters = np.array([])
 
 
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
    
    D = LinearOperator( (N,N), matvec=DS)
    xDS, exitCode = gmres(D,b)
    print("DS Result: ",xDS)
    T = LinearOperator( (N,N), matvec=TC)
    xTC, exitCode = gmres(T,b)
    print("TC Result: ",xTC)
    print("Difference: ", xDS-xTC)
    

    
    
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
