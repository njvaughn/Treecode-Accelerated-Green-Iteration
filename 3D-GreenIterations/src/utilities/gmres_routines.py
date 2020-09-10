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
        self.residuals = []
    def __call__(self, rk=None):
        self.niter += 1
        self.residuals.append(rk)
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
            
class gmres_counter_err(object):
    def __init__(self, ref_sol, disp=True):
        self._disp = disp
        self.niter = 0
        self.sol = None
        self.errors=[]
        self.ref_sol = ref_sol
    def __call__(self, x=None):
        self.sol = np.copy(x)
        self.niter += 1
        if self._disp:
            print('iter %3i' % (self.niter))
        
        error = np.linalg.norm(self.sol - self.ref_sol)
        self.errors.append(error)
        


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
                    
#         if cellIdx==0:
# #             print(block_diag[cellIdx,:,:])
#             print("cell 0 diagonal: ")
#             print(np.diag(block_diag[cellIdx,:,:]) )
                    
                    
                    
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
        
        
        
        if singularity==BT.Singularity.SKIPPING:
            # if skipping, W array doesn't get used.  Need to pre-multiply.
            charge = RHO*Ws
        elif singularity==BT.Singularity.SUBTRACTION:
            charge = RHO
        else: 
            print("What is singularity? ", singularity)
            exit(-1)
            
        
        output = BT.callTreedriver(  Nt, Ns,
                                 np.copy(Xt), np.copy(Yt), np.copy(Zt), np.copy(charge),
                                 np.copy(Xs), np.copy(Ys), np.copy(Zs), np.copy(charge), np.copy(Ws),
                                 kernel, numberOfKernelParameters, kernelParameters,
                                 singularity, approximation, computeType,
                                 GPUpresent, treecode_verbosity, 
                                 theta=theta, degree=treecodeDegree, sourceLeafSize=maxPerSourceLeaf, targetLeafSize=maxPerTargetLeaf, sizeCheck=1.0)
        

        return output
    
    return treecode

def treecode_closure_Veff(Nt, Ns,
                     Xt, Yt, Zt, Vt,
                     Xs, Ys, Zs, Vs, Ws,
                     kernel, numberOfKernelParameters, kernelParameters,
                     singularity, approximation, computeType,
                     GPUpresent, treecode_verbosity, 
                     theta, treecodeDegree, maxPerSourceLeaf, maxPerTargetLeaf):
    
    def treecode(psi):
        
        # Apply A*psi = (I+GV)*psi
        output = BT.callTreedriver(  Nt, Ns,
                                 np.copy(Xt), np.copy(Yt), np.copy(Zt), np.copy(psi)*Vt,
                                 np.copy(Xs), np.copy(Ys), np.copy(Zs), np.copy(psi)*Vs, np.copy(Ws),
                                 kernel, numberOfKernelParameters, kernelParameters,
                                 singularity, approximation, computeType,
                                 GPUpresent, treecode_verbosity, 
                                 theta=theta, degree=treecodeDegree, sourceLeafSize=maxPerSourceLeaf, targetLeafSize=maxPerTargetLeaf, sizeCheck=1.0)
        
        y = psi + 1/2/np.pi*output
                                   
        return y
    
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
        y = psi + -1/2/np.pi * BT.callTreedriver(  Nt, Ns,
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


def GMRES(A,b,x0,w=[1], e=None, nmax_iter=3):
    
#     if not x0:
#         x0 = np.ones_like(b)
    
    if len(w)==1:
        w = np.ones_like(b)
    
    
    r = b - np.asarray(np.dot(A, x0*w)).reshape(-1)

    x = []
    q = [0] * (nmax_iter)

    x.append(r)

    q[0] = r / np.sqrt(np.dot(r, r*w))
#     q[0] = r / np.linalg.norm(r)

    h = np.zeros((nmax_iter + 1, nmax_iter))

    for k in range(min(nmax_iter, A.shape[0])):
#         print("k = %i" %k)
        
        y = np.asarray(np.dot(A, q[k]*w)).reshape(-1)
        

        for j in range(k+1):
            h[j, k] = np.dot(q[j], y*w)
            y = y - h[j, k] * q[j]
            
#         h[k + 1, k] = np.sqrt(np.dot(y, y*w)) 
        h[k + 1, k] = np.linalg.norm(y) 
        if (h[k + 1, k] != 0 and k != nmax_iter - 1):
            q[k + 1] = y / h[k + 1, k]

        b = np.zeros(nmax_iter + 1)
        b[0] = np.sqrt(np.dot(r, r*w)) 
#         b[0] = np.linalg.norm(r) 

        result = np.linalg.lstsq(h, b)[0]

        x.append(np.dot(np.asarray(q).transpose(), result) + x0)

    return x


def GMRES_PC(A,M,b,x0,w=[1], e=None, nmax_iter=3):
    
#     if not x0:
#         x0 = np.ones_like(b)
    
    if len(w)==1:
        w = np.ones_like(b)
    
    
    r = b - np.asarray(np.dot(M,np.dot(A, x0*w))).reshape(-1)

    x = []
    q = [0] * (nmax_iter)

    x.append(r)

    q[0] = r / np.sqrt(np.dot(r, r*w))
#     q[0] = r / np.linalg.norm(r)

    h = np.zeros((nmax_iter + 1, nmax_iter))

    for k in range(min(nmax_iter, A.shape[0])):
#         print("k = %i" %k)
        
        y = np.asarray(np.dot(M,np.dot(A, q[k]*w))).reshape(-1)
        

        for j in range(k+1):
            h[j, k] = np.dot(q[j], y*w)
            y = y - h[j, k] * q[j]
            
#         h[k + 1, k] = np.sqrt(np.dot(y, y*w)) 
        h[k + 1, k] = np.linalg.norm(y) 
        if (h[k + 1, k] != 0 and k != nmax_iter - 1):
            q[k + 1] = y / h[k + 1, k]

        b = np.zeros(nmax_iter + 1)
        b[0] = np.sqrt(np.dot(r, r*w)) 
#         b[0] = np.linalg.norm(r) 

        result = np.linalg.lstsq(h, b)[0]

        x.append(np.dot(np.asarray(q).transpose(), result) + x0)

    return x

def iterative_GMRES(A,b,x0):
    counter = gmres_counter()
    x = np.copy(x0)
    alpha=1
    nmax_iter=200
    
    A = A+np.diag(20*np.ones_like(x0))


    res=1
    count=1
    while ( (res>1e-5) and (count<20) ):
    
        c = b+alpha*x
        sol, exitcode1 = la.gmres(A,c,x0=x, maxiter=1, restart=nmax_iter, atol=1e-10)
        
        res = np.linalg.norm(sol-x)
        print("outer iteration %i, res = %1.3e" %(count,res) )
        
        count+=1
        x = np.copy(sol)
    
    
    return x
    

if __name__=="__main__":
    
    
#     numCells=3
#     quadratureOrder=1
#     N = numCells * (quadratureOrder+1)**3
#     print("N = ", N)
    np.random.seed(1)
    N = 1000

    X = np.random.rand(N)
    Y = np.random.rand(N)
    Z = np.random.rand(N)
    W = np.random.rand(N)
#     Q = np.random.rand(N)
    
    W = np.ones(N)/N
#     W = np.ones(N)
#     Q = np.ones(N)
    
    
#     print("Q = ", Q)

#     nmax_iter=100
    nmax_iter=N
    
    ## Direct Sum Matrix
#     A = np.zeros((N,N))
#     B = np.zeros((N,N))
#     for i in range(N):
#         for j in range(N):
#             if i!=j:
#                 dx = X[i]-X[j]
#                 dy = Y[i]-Y[j]
#                 dz = Z[i]-Z[j]
#                 r = np.sqrt(dx*dx + dy*dy + dz*dz)
#                 A[i,j] = 1/r
#             else:
#                 A[i,i] = 0
#       
#     b = np.ones(N)
#     sol_nonreg = np.linalg.solve(A,b)
                
    ## Regularized Direct Sum Matrix
    A = np.zeros((N,N))
    B = np.zeros((N,N))
    epsilon=0.1
    for i in range(N):
        for j in range(N):
            dx = X[i]-X[j]
            dy = Y[i]-Y[j]
            dz = Z[i]-Z[j]
#             r = np.sqrt(dx*dx + dy*dy + dz*dz + epsilon*epsilon)
            r = np.sqrt(dx*dx + dy*dy + dz*dz)
#             A[i,j] = Q[j]*W[j]/r
            A[i,j] = W[j]/np.sqrt(r*r + epsilon*epsilon)
            
            if i==j:
                B[i,i] = W[j]/np.sqrt(r*r + epsilon*epsilon)
#             A[i,j] = Q[j]*W[j]*r/(r*r + epsilon*epsilon)
            
                
#                 if r<1.3:
#                     B[i,j] = Q[j]*W[j]/r
#                     
    Binv = np.linalg.inv(B)
                
#     A = np.zeros((N,N))
#     for i in range(N):
#         print("i=",i)
#         if i>0:
#             A[i,i-1] = -1
#         A[i,i]   =  2
#         if i<N-1:
#             A[i,i+1] = -1

                
#     A = np.random.rand(N,N) + 20*np.diag(np.ones(N))
# 
#     A = A+A.transpose()
#     b = np.random.rand(N)
    b = np.ones(N)
    x0 = np.ones(N)
    
    
    sol = np.linalg.solve(A,b)
    
#     reg_error = np.linalg.norm(sol-sol_nonreg)/np.linalg.norm(sol)
#     print("Regularization error = %1.3e" %reg_error)
    
    ## ATTEMPT AT ITERATIVE SOLVE USING SHIFT
#     it_sol = iterative_GMRES(A,b,x0)
#     
#     error = np.linalg.norm(sol-it_sol)
#     print("Error from iterative GMRES with shift: ", error)
#     exit(-1)

#     x0 = sol+0.1*np.ones(N)
    
#     start = time.time()
#     x = GMRES(A,b,x0=x0,nmax_iter=nmax_iter)
# #     x = np.zeros_like(sol)
#     manual_time=time.time()-start
#     
# #     print("\nDirect solve solution = ", sol)
# #     print()
#     errors=[]
#     for i in range(min(nmax_iter,N)+1):
# #         print("i = ", i)
# #         print("solution = ", x[i])
#         error=np.linalg.norm(x[i]-sol)/np.linalg.norm(sol)
#         errors.append(error)
#         print("iteration %2i, error = %1.3e" %(i, error ) )
# #         print()
# #     print(A.dot(sol) - b)
    
    counter1=gmres_counter()
    counter2=gmres_counter()
    counter_err = gmres_counter_err(sol)
    counter_x=gmres_counter_x()
    
    start=time.time()
#     sp_sol1, exitcode1 = la.gmres(A,b,x0, maxiter=1, restart=nmax_iter, tol=1e-14)
    sp_sol1, exitcode1 = la.gmres(A,b,x0=x0, maxiter=1, restart=nmax_iter, tol=1e-6)
    sp_time1 = time.time()-start
    start=time.time()
#     sp_sol2, exitcode2 = la.gmres(A,b,x0, M=Binv, maxiter=1, restart=nmax_iter, tol=1e-14)
    sp_sol2, exitcode2 = la.gmres(A,b,x0=x0, M=Binv, maxiter=1, restart=nmax_iter, tol=1e-6)
#     sp_sol2, exitcode2 = la.gmres(A,b,x0, callback_type='x', callback=counter_err, maxiter=1, restart=nmax_iter, tol=1e-13)
    sp_time2=time.time()-start
    
    
#     sp_sol2, exitcode2 = la.gmres(A,b,x0, callback_type='pr_norm', callback=counter1, maxiter=nmax_iter, restart=1, tol=1e-16)  # iterations count the number of restarts.  
#     sp_sol3, exitcode3 = la.gmres(A,b,x0, M=Binv, callback_type='pr_norm', callback=counter2, maxiter=1, restart=nmax_iter, atol=1e-16)  # iterations count the number of restarts.  
    print("exit codes: ", exitcode1, exitcode2)
#     print("x0               = ", x0)
#     print("manual gmres sol = ", x[-1])
#     print("sp_sol1          = ", sp_sol1)
#     print("sp_sol2          = ", sp_sol2)
    
#     print("scipy gmres: ", sp_sol, exitcode)
#     print("\nmanual erorr   = %1.3e, time = %1.1e s" %(np.linalg.norm(x[-1]-sol)/np.linalg.norm(sol) ,manual_time))
    print("\nscipy erorr 1  = %1.3e, time = %1.1e s" %(np.linalg.norm(sp_sol1-sol)/np.linalg.norm(sol), sp_time1 ))
    print("\nscipy erorr 2  = %1.3e, time = %1.1e s" %(np.linalg.norm(sp_sol2-sol)/np.linalg.norm(sol), sp_time2 ))
    print("\nDifference between PC and non-PC = %1.3e" %(np.linalg.norm(sp_sol1-sp_sol2)))
#     print("\nscipy erorr 3  = %1.3e" %(np.linalg.norm(sp_sol3-sol)/np.linalg.norm(sol) ))
    
    
    import matplotlib.pyplot as plt
    
    evals = np.linalg.eigvals(A)
    evals_PC = np.linalg.eigvals(np.dot(Binv,A))
    
    reals = np.real(evals)
    tempMax1 = np.max(np.abs(reals))
    
    imags = np.imag(evals)
    tempMax2 = np.max(np.max(np.abs(reals)))
    
    bounds = 1.1*max(tempMax1, tempMax2)
    
#     print("eigenvalues = ", evals)
#     plt.figure()


#     # Print Errors and Eigenvalues...
#     
#     fig,(ax1,ax2) = plt.subplots(1,2, figsize=(8,4))
#     ax1.semilogy(errors,'bo-')
#     ax1.semilogy(counter_err.errors,'ro-')
#     ax1.set_xlabel("GMRES Iteration")
#     ax1.set_ylabel("Error")
#     ax1.set_title("GMRES")
#      
#     ax2.plot(np.real(evals_PC), np.imag(evals_PC), 'rx')
#     ax2.plot(np.real(evals), np.imag(evals), '.')
#     ax2.set_xlabel("real component")
#     ax2.set_ylabel("imaginary component")
#     ax2.set_title("Eigenvalues")
#     ax2.set_aspect('equal')
#     ax2.set_ylim([-bounds,bounds])
#     ax2.set_xlim([-bounds,bounds])
# #     fig.suptitle("GMRES for random matrix of size %i x %i" %(N,N))
# #     fig.suptitle("GMRES for SHIFTED random matrix of size %i x %i" %(N,N))
# #     fig.suptitle("GMRES for finite difference matrix of size %i x %i" %(N,N))
# #     fig.suptitle("GMRES for %i randomly distributed particles" %N)
#     fig.suptitle("%i randomly distributed particles, regularized Coulomb eps=%1.2f" %(N,epsilon))
# #     fig.suptitle("GMRES for %i randomly distributed particles WITH DIAG" %N)
#     plt.tight_layout()
#     plt.subplots_adjust(top=0.85)
#     plt.show()
     
    
# #     ## PRECONDITIONER EIGENVALUES
#     fig,(ax1) = plt.subplots(1,1, figsize=(8,3))
#     ax1.plot(np.real(evals_PC), np.imag(evals_PC), 'rx', label='Diagonal Preconditioner')
#     ax1.plot(np.real(evals), np.imag(evals), '.', label="No Preconditioner")
#     ax1.set_xlabel("real component")
#     ax1.set_ylabel("imaginary component")
#     ax1.set_title("Eigenvalues, epsilon=%1.2f" %epsilon)
# #     ax1.set_aspect('equal')
#     ax1.set_ylim([-bounds,bounds])
#     ax1.set_xlim([-bounds,bounds])
#     plt.legend()
#     plt.tight_layout()
#     plt.show()
     
    
     
#     block_diag = create_block_diagonal(X, Y, Z, W, quadratureOrder, numCells)
#     inv_block_diag = invert_block_diagonal(block_diag)
#     verify_inverse(block_diag, inv_block_diag)
    
#     apply_inv_block_diagonal(X, Y, Z, W, quadratureOrder, numCells, q)
    
    
    
    
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
