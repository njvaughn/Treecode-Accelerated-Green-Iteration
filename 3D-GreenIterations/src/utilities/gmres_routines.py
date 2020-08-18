import numpy as np
import scipy as sp
from scipy.sparse.linalg import gmres
from scipy.sparse.linalg import LinearOperator
import time

import scipy.sparse.linalg as la

import BaryTreeInterface as BT
from meshUtilities import GlobalLaplacian


class gmres_counter(object):
    def __init__(self, disp=True):
        self._disp = disp
        self.niter = 0
    def __call__(self, rk=None):
        self.niter += 1
        if self._disp:
            print('iter %3i\trk = %s' % (self.niter, str(rk)))
            
class gmres_counter_x(object):
    def __init__(self, disp=True):
        self._disp = disp
        self.niter = 0
        self.sol = None
    def __call__(self, x=None):
        self.sol = np.copy(x)
        self.niter += 1
        if self._disp:
            print('iter %3i' % (self.niter))


def matrix_free(x):
    b = np.zeros_like(x)
    b[0] = 1*x[0] + 2*x[1]
    b[1] = 2*x[0] + 3*x[1]
    return b


def create_block_diagonal(X, Y, Z, W, quadratureOrder, numCells, alpha):
    
    ptsPerCell = (quadratureOrder+1)**3
    block_diag = np.zeros([numCells,ptsPerCell,ptsPerCell])
 
    
    for cellIdx in range(numCells):
        startingIdx = ptsPerCell*cellIdx
        for i in range(ptsPerCell):
            
            xi = X[startingIdx+i]
            yi = Y[startingIdx+i]
            zi = Z[startingIdx+i]
            
            # Intialize diagonal to the added correction term
            block_diag[cellIdx,i,i] = (2*np.pi*alpha*alpha) 
            
            for j in range(ptsPerCell):
                
                if j!=i:
                    dx = xi - X[startingIdx+j]
                    dy = yi - Y[startingIdx+j]
                    dz = zi - Z[startingIdx+j]
                    wj =      W[startingIdx+j]
                    
                    
                    rij = np.sqrt( dx*dx + dy*dy + dz*dz )
                    
                    # Fill In Off-Diagonal (i,j)
                    block_diag[cellIdx,i,j] = wj/rij
                    
                    
                    # Increment Diagonal (i,i)
                    block_diag[cellIdx,i,i] -= np.exp(-rij*rij/(alpha*alpha))*wj/rij
                    
        if cellIdx==0:
#             print(block_diag[cellIdx,:,:])
            print("cell 0 diagonal: ")
            print(np.diag(block_diag[cellIdx,:,:]) )
                    
                    
                    
#                     if not gaussian:
#                         block_diag[cellIdx,i,j] = wj/rij
#                     else:
#                         block_diag[cellIdx,i,j] = np.exp(-alpha**2*rij*rij)*wj/rij
#                 else: 
#                     if not gaussian:
#                         block_diag[cellIdx,i,j] = (2*np.pi*alpha*alpha) 
#                     else:
#                         block_diag[cellIdx,i,j] = 0.0
        
    return block_diag

def invert_block_diagonal(block_diag, verify=False):
    
    numCells, ptsPerCell, _ = np.shape(block_diag)
    
    inv_block_diag = np.empty([numCells,ptsPerCell,ptsPerCell])
    
    if verify:
        assert np.shape(inv_block_diag) == np.shape(block_diag), "error: shape of block diagonal not equal to shape of inverse"
    
    for cellIdx in range(numCells):
        
        block = block_diag[cellIdx,:,:]
        inv = np.linalg.inv(block)
        inv_block_diag[cellIdx,:,:] = inv
            
    return inv_block_diag

def verify_inverse(block_diag, inv_block_diag):
    
    numCells, ptsPerCell, _ = np.shape(block_diag)
    
    for cellIdx in range(numCells):
        A = block_diag[cellIdx,:,:]
        Ainv = inv_block_diag[cellIdx,:,:]
        
        Prod = np.matmul(A,Ainv)
        
        for i in range(ptsPerCell):            
            for j in range(ptsPerCell):
                if i==j:
                    assert abs(1-Prod[i,j]) < 1e-12, "Diagonal not close enough to 1"
                else:
                    assert abs(Prod[i,j]) < 1e-12, "Off-diagonal not close enough to 0"
    
    print("[verification] Inverse computed correctly.")
    return
#                 if abs(Prod[i,j])<1e-12:
#                     Prod[i,j] = 0
                
        
#         print("cell %i, product = " %cellIdx, Prod)
        
        
    
    

def inv_block_diagonal_closure(X, Y, Z, W, alpha, quadratureOrder, numCells, verify=False):
    
    block_diag = create_block_diagonal(X, Y, Z, W, quadratureOrder, numCells, alpha)
    inv_block_diag = invert_block_diagonal(block_diag)
    
#     block_diag_noAlpha = create_block_diagonal(X, Y, Z, W, quadratureOrder, numCells, alpha=alpha, gaussian=True)
#     inv_block_diag_noAlpha = invert_block_diagonal(block_diag_noAlpha)
    
    if verify:
        verify_inverse(block_diag, inv_block_diag)
    
    def inv_block_diagonal(q):
        
        
        
        _, ptsPerCell, _ = np.shape(block_diag)
        
        y = np.empty_like(q)
        
        # Compute y = Ainv*q
        for cellIdx in range(numCells):
            
            startingIdx = cellIdx*ptsPerCell
            Ainv = inv_block_diag[cellIdx,:,:]
            x = q[startingIdx:startingIdx+ptsPerCell]
            y[startingIdx:startingIdx+ptsPerCell] = Ainv.dot(x)
            
            
        
#         # Compute y = Ainv*q
#         for cellIdx in range(numCells):
#             startingIdx = cellIdx*ptsPerCell
#             A = inv_block_diag_alpha[cellIdx,:,:]
# #             B = inv_block_diag_noAlpha[cellIdx,:,:]
#             x = q[startingIdx:startingIdx+ptsPerCell]
#             
# #             temp = B*x
# #             ones = np.ones_like(x)
#             y[startingIdx:startingIdx+ptsPerCell] = A.dot(x)
# #             y[startingIdx:startingIdx+ptsPerCell] = A.dot(x) - temp.dot(ones)
# #             y[startingIdx:startingIdx+ptsPerCell] = B.dot(x)
#             
# #             print("shapes: ", np.shape(A.dot(x)), np.shape(x.dot(A)))
        
        if verify:
            
            qq = np.empty_like(q)
            # Compute qq = A*y, check qq close to q
            for cellIdx in range(numCells):
                startingIdx = cellIdx*ptsPerCell
                A = block_diag[cellIdx,:,:]
                x = y[startingIdx:startingIdx+ptsPerCell]
                
                qq[startingIdx:startingIdx+ptsPerCell] = A.dot(x)
                
            error = np.sqrt( np.sum( (q-qq)**2 ))
            assert error < 1e-13, "recovered q not accurate enough, error = %1.3e" %error
            print("Inverse verification passed.")
        
        return y
    return inv_block_diagonal


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
        
        output = BT.callTreedriver(  Nt, Ns,
                                 np.copy(Xt), np.copy(Yt), np.copy(Zt), np.copy(RHO),
                                 np.copy(Xs), np.copy(Ys), np.copy(Zs), np.copy(RHO), np.copy(Ws),
                                 kernel, numberOfKernelParameters, kernelParameters,
                                 singularity, approximation, computeType,
                                 GPUpresent, treecode_verbosity, 
                                 theta=theta, degree=treecodeDegree, sourceLeafSize=maxPerSourceLeaf, targetLeafSize=maxPerTargetLeaf, sizeCheck=1.0)
                                   
        return output
    
    return treecode



def D_closure(   Nt, Ns,
                 Xt, Yt, Zt, Vt,
                 Xs, Ys, Zs, Vs, Ws,
                 kernel, numberOfKernelParameters, kernelParameters,
                 singularity, approximation, computeType,
                 GPUpresent, treecode_verbosity, 
                 theta, treecodeDegree, maxPerSourceLeaf, maxPerTargetLeaf,
                 DX_matrices, DY_matrices, DZ_matrices, order):
    
    def D_operator(psi):
        
#         Dv = B{-1}Av
        
        
#         Av = (I+GV)v
        y = psi + BT.callTreedriver(  Nt, Ns,
                                 np.copy(Xt), np.copy(Yt), np.copy(Zt), Vt*psi,
                                 np.copy(Xs), np.copy(Ys), np.copy(Zs), Vs*psi, np.copy(Ws),
                                 kernel, numberOfKernelParameters, kernelParameters,
                                 singularity, approximation, computeType,
                                 GPUpresent, treecode_verbosity, 
                                 theta=theta, degree=treecodeDegree, sourceLeafSize=maxPerSourceLeaf, targetLeafSize=maxPerTargetLeaf, sizeCheck=1.0)
        
#         B^{-1}v = (-1/4pi) Delta v
        x = GlobalLaplacian( DX_matrices, DY_matrices, DZ_matrices, y, order)
        x /= -4*np.pi
                                   
        return x
    
    return D_operator

def H_closure( Veff, DX_matrices, DY_matrices, DZ_matrices, order):
    
    def H_operator(psi):
        
        x = -1/2 * GlobalLaplacian( DX_matrices, DY_matrices, DZ_matrices, psi, order) + Veff*psi 
#         x = GlobalLaplacian( DX_matrices, DY_matrices, DZ_matrices, psi, order)  
                
        return x
    
    return H_operator


if __name__=="__main__":
    
    
    numCells=3
    quadratureOrder=1
    N = numCells * (quadratureOrder+1)**3
    print("N = ", N)
    X = np.random.rand(N)
    Y = np.random.rand(N)
    Z = np.random.rand(N)
    W = np.random.rand(N)
    q = np.random.rand(N)
    
     
#     block_diag = create_block_diagonal(X, Y, Z, W, quadratureOrder, numCells)
#     inv_block_diag = invert_block_diagonal(block_diag)
#     verify_inverse(block_diag, inv_block_diag)
    
    apply_inv_block_diagonal(X, Y, Z, W, quadratureOrder, numCells, q)
    
    
    
    
#     print("running gmres_routines.py")
#     
#     N = 30
#     X = np.random.rand(N)
#     Y = np.random.rand(N)
#     Z = np.random.rand(N)
# #     
# #     RHO  = np.ones(N)
# #     
# #     b = np.ones(N)
# #     
# #     DS=direct_sum_closure(x,y,z)
# #     A = LinearOperator( (N,N), matvec=DS)
# #     x2, exitCode = gmres(A,b)
# #     print("Result: ",x2)
# #     
# #     for i in range(N):
# #         print(x[i],y[i],z[i])
# #         
#         
#     # set treecode parameters
#     N=120
#     maxPerSourceLeaf = 5
#     maxPerTargetLeaf = 5
#     GPUpresent = False
#     theta = 0.8
#     treecodeDegree = 3
#     gaussianAlpha = 1.0
#     verbosity = 0
#      
#     approximation = BT.Approximation.LAGRANGE
#     singularity   = BT.Singularity.SUBTRACTION
#     computeType   = BT.ComputeType.PARTICLE_CLUSTER
#      
#     kernel = BT.Kernel.COULOMB
#     numberOfKernelParameters = 1
#     kernelParameters = np.array([1.0])
#  
#  
#     # initialize some random data
#     np.random.seed(1)
#     RHO = np.random.rand(N)
#     X = np.random.rand(N)
#     Y = np.random.rand(N)
#     Z = np.random.rand(N)
#     W = np.ones(N)   # W stores quadrature weights for convolution integrals.  For particle simulations, set = ones.
#     
#     DS=direct_sum_closure(X,Y,Z)
#     
#     
#     TC=treecode_closure( N, N,
#                      np.copy(X), np.copy(Y), np.copy(Z),
#                      X, Y, Z, W,
#                      kernel, numberOfKernelParameters, kernelParameters,
#                      singularity, approximation, computeType,
#                      GPUpresent, verbosity, 
#                      theta, treecodeDegree, maxPerSourceLeaf, maxPerTargetLeaf)
#     
#     
#     
#     b = np.random.rand(N)
#     
# #     D = LinearOperator( (N,N), matvec=DS)
# #     counterDS = gmres_counter()
# #     xDS, exitCode = gmres(D,b,callback=counterDS)
# #     print("DS Result: ",xDS)
#     T = LinearOperator( (N,N), matvec=TC)
#     counterTC = gmres_counter(disp=False)
#     counterTC2 = gmres_counter(disp=False)
#     xTC1, exitCode = la.lgmres(T,b,callback=counterTC,tol=1e-5)
#     
#     
#     
#     gmresStart=time.time()
# #     xTC1, exitCode = la.gmres(T,b,callback=counterTC,tol=1e-6, maxiter=5000)
#     xTC1, exitCode = la.lgmres(T,b,callback=counterTC,tol=1e-5)
#     gmresStop=time.time()
#     print(" GMRES took %f seconds and %i iterations." %(gmresStop-gmresStart, counterTC.niter) )
#     lgmresStart=time.time()
# #     xTC2, exitCode = la.lgmres(T,b,callback=counterTC2,tol=1e-5)
#     xTC2, exitCode = la.minres(T,b,callback=counterTC2,tol=1e-5)
#     lgmresStop=time.time()
# #     print("TC Result: ",xTC)
# #     print("Difference: ", xTC1-xTC2)
#     normDiff = np.sqrt( np.sum( xTC1-xTC2)**2 )
#     print("LGMRES took %f seconds and %i iterations." %(lgmresStop-lgmresStart, counterTC2.niter) )
#     print("L2 norm of difference: ", normDiff)
# #     print(counterTC2)
#     
# 
#     
#     
# #     A = np.array([[1,2],[2,3]])
# #     print(A)
# #     b  = [1,-1]
# #     x0 = [1,1]
# #     
# #     x1, exitCode = gmres(A,b,x0)
# #     print(x1)
# #     
# #     A = LinearOperator( (2,2), matvec=matrix_free)
# #     x2, exitCode = gmres(A,b,x0)
# #     print(x2)
