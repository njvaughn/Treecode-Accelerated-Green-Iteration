import numpy as np
import scipy as sp
import time
import scipy.sparse.linalg as sla
import numpy.linalg as nla

import BaryTreeInterface as BT
from gmres_routines import gmres_counter, gmres_counter_x



def power_iteration_A(A,B,x,referenceEigenvalue=-1.76758271):
    
    count=1
    residual=2
    eigenvalue=1
    z_old = np.copy(x) / nla.norm(x)
    
    
    while ( (residual>1e-5) and (count<200) ):
        
        # Apply A
        y = A.dot(z_old)
        
        # Solve Binv
        z_new = nla.solve(B,y)
        counter = gmres_counter()
#         z_new_gmres, exitcode = sla.gmres(B,y,x0=eigenvalue*z_old, callback=counter,maxiter=10)
        z_new_gmres, exitcode = sla.gmres(B,y,x0=eigenvalue*z_old,maxiter=10)
         
        gmres_error = nla.norm(z_new-z_new_gmres)/nla.norm(z_new)
        
        
        z_new_norm = nla.norm(z_new)
        z_new /= z_new_norm
        
        
        if nla.norm(z_old-z_new) > nla.norm(z_old+z_new):
            eigenvalue = -z_new_norm
        else:
            eigenvalue =  z_new_norm
        

        residual = min( nla.norm(z_old-z_new), nla.norm(z_old+z_new) )  # account for positive or negative eigenvalues
        print("count %2i: Residual = %1.3e, Rayleigh quotient  = %1.8f, error = %1.3e" %(count, residual, eigenvalue, eigenvalue-referenceEigenvalue))
        count+=1
        z_old=np.copy(z_new)
        
def power_iteration_A_gmres(A,B,x,referenceEigenvalue=-1.76758271):
    
    N=len(x)
    count=1
    residual=2
    eigenvalue=2
    z_old = np.copy(x) / nla.norm(x)
    
    
#     M_x = lambda x: sla.spsolve(B, x)
#     M = sla.LinearOperator((N, N), M_x)

#     M = np.diag(B)
    M_x = np.diag(np.linalg.inv(B))
    M = np.diag(M_x)
    
#     print("M = ", M)
    
    adaptpive_tol=0.25
    factor=1.02
    
    counter = gmres_counter(disp=False)
    counter_x = gmres_counter_x()
    
    while ( (residual>1e-8) and (count<50) ):
        
        # Apply A
        y = A.dot(z_old)
        
        # Solve Binv
#         z_new = nla.solve(B,y)
        
        
#         z_new_gmres, exitcode = sla.gmres(B,y,x0=eigenvalue*z_old, callback=counter,maxiter=10)
#         z_new, exitcode = sla.gmres(B,y,x0=np.ones(N), callback=counter, maxiter=7, tol=0.2)
#         z_new, exitcode = sla.gmres(B,y,x0=eigenvalue*z_old, callback=counter, maxiter=9)
#         z_new, exitcode = sla.gmres(B,y,x0=eigenvalue*z_old, callback=counter, maxiter=9)
#         print("znew = ", z_new)
#         print("exitcode = ", exitcode)
        numIterations=30
        z_new, exitcode = sla.gmres(B,y, x0=eigenvalue*z_old, callback=counter, maxiter=numIterations, restart=numIterations)
#         z_new, exitcode = sla.gmres(B,y, callback=counter_x, callback_type='x', maxiter=2, restart=5)
#         z_new, exitcode = sla.gmres(B,y,x0=eigenvalue*z_old, callback=counter, maxiter=9, restart=50, tol=adaptpive_tol)
#         z_new, exitcode = sla.gmres(B,y,x0=eigenvalue*z_old, callback=counter, maxiter=11, restart=50, tol=1e-10)
#         adaptpive_tol/=factor
#         if count>3:
#             factor = factor*factor
#         adaptpive_tol = max(1e-10,adaptpive_tol)
#         print("adaptive tolerance = %1.3e" %adaptpive_tol)
#         z_new = counter_x.sol
#         z_new, exitcode = sla.gmres(B,y,M=M,x0=eigenvalue*z_old, callback=counter, maxiter=10)
#         z_new, exitcode = sla.gmres(B,y,maxiter=20)
         
#         gmres_error = nla.norm(z_new-z_new_gmres)/nla.norm(z_new)
#         print("znew = ", z_new)        
#         print("znew-counter.sol = ", z_new-counter_x.sol)        
#         print("exitcode = ", exitcode)
        
        
        z_new_norm = nla.norm(z_new)
        z_new /= z_new_norm
        
        
        if nla.norm(z_old-z_new) > nla.norm(z_old+z_new):
            eigenvalue = -z_new_norm
        else:
            eigenvalue =  z_new_norm
        

        residual = min( nla.norm(z_old-z_new), nla.norm(z_old+z_new) )  # account for positive or negative eigenvalues
        print("count %2i (GMRES %3i): Residual = %1.3e, Rayleigh quotient  = %1.8f, error = %1.3e" %(count, counter_x.niter, residual, eigenvalue, eigenvalue-referenceEigenvalue))
        count+=1
        z_old=np.copy(z_new)
        
def power_iteration_B(A,B,x,referenceEigenvalue=-1.76758271):
    
    count=1
    residual=2
    z_old = np.copy(x) / nla.norm(x)
    
    
    while ( (residual>1e-5) and (count<50) ):
        
        # Apply A
        y = B.dot(z_old)
        
        # Solve Binv
        z_new = nla.solve(A,y)
        
        
        z_new_norm = nla.norm(z_new)
        z_new /= z_new_norm
        
        
        if nla.norm(z_old-z_new) > nla.norm(z_old+z_new):
            eigenvalue = -1/z_new_norm
        else:
            eigenvalue =  1/z_new_norm
        

        residual = min( nla.norm(z_old-z_new), nla.norm(z_old+z_new) )  # account for positive or negative eigenvalues
        print("count %2i: Residual = %1.3e, Rayleigh quotient  = %1.8e, error = %1.3e" %(count, residual, eigenvalue, eigenvalue-referenceEigenvalue))
        count+=1
        z_old=np.copy(z_new)
        


if __name__=="__main__":
    
    # Routines for solving the generalized eigenvalue problem of the form A*x=lambda*B*x
    
    # (1) convert to standard eigenvalue problem D*x = (B^-1)*A*x = lambda*x
    # (2) apply operator D in two steps:   z = D*x
    #     (a) compute intermediate y       y = A*x
    #     (b) solve linear system        B*z = y
    
    # A and B opertaors are G and (I+GV), where G is the convolution with the coulomb kernel.  A and B can be arranged to be either one.
    
    N=30
    np.random.seed(1)
    A = np.random.rand(N,N)/ N 
    B = np.random.rand(N,N) #+ np.diag(np.ones(N))
    
    Binv = nla.inv(B)
    
    D = np.matmul(Binv, A)
    direct_eigenvalues, direct_eigenvectors = sla.eigs(D,k=N)
    
    maxId = np.argmax(np.abs(direct_eigenvalues))
    minId = np.argmin(np.abs(direct_eigenvalues))
    
    
    print("Direct Eigenvalues: ", direct_eigenvalues)
    
    
    init_guess = np.random.rand(N)
    power_iteration_A_gmres(A,B,init_guess,referenceEigenvalue=direct_eigenvalues[maxId])
#     power_iteration_A(A,B,init_guess,referenceEigenvalue=direct_eigenvalues[maxId])
#     power_iteration_B(A,B,init_guess,referenceEigenvalue=direct_eigenvalues[minId])
    
    
    
# #     print(A)
# #     print(sla.eig(A,k=2))
#     print(D)
#     print(eigenvalues)
#     print()
#     print(np.shape(eigenvectors))
#     print(eigenvectors[:,0])
#     print(eigenvectors[0])
#     print()
#     
#     out = np.matmul(D, eigenvectors[:,0])
# #     out2 = D.dot(eigenvectors[0])
#     print(out)
#     print(out/eigenvalues[0])
#     
#     error = eigenvectors[:,0] - out/eigenvalues[0]
#     print(error)
# #     print(out2)



    
    
#     print("testing")