import numpy as np
import matplotlib.pyplot as plt
from numba import jit, njit, cuda
from math import sqrt
import time 

from mpiUtilities import global_dot

# @jit(parallel=True)
def modifiedGramSchmidt(V, weights, comm):
    k,n = np.shape(V)
    U = np.zeros_like(V)
    U[0,:] = V[0,:] / np.sqrt( global_dot(V[0,:],V[0,:]*weights, comm) )
    for i in range(1,k):
        U[i,:] = V[i,:]
        for j in range(i):
#             rprint(rank,'Orthogonalizing %i against %i' %(i,j))
            U[i,:] -= (global_dot(U[i,:],U[j,:]*weights, comm) / global_dot(U[j,:],U[j,:]*weights, comm))*U[j,:]
        U[i,:] /= np.sqrt( global_dot(U[i,:],U[i,:]*weights, comm) )
        
    return U



# @jit()
def modifiedGramSchmidt_singleOrbital(V,weights,targetOrbital, comm):
    U = V[:,targetOrbital]
    for j in range(targetOrbital):
#         print('Orthogonalizing %i against %i' %(targetOrbital,j))
#         U -= (np.dot(V[:,targetOrbital],V[:,j]*weights) / np.dot(V[:,j],V[:,j]*weights))*V[:,j]
#         U -= np.dot(V[:,targetOrbital],V[:,j]*weights) *V[:,j]
#         U /= np.sqrt( np.dot(U,U*weights) )
        
        U -= global_dot(V[:,targetOrbital],V[:,j]*weights, comm) *V[:,j]
        U /= np.sqrt( global_dot(U,U*weights, comm) )
    
#     U /= np.sqrt( np.dot(U,U*weights) )  # normalize again at end (safegaurd for the zeroth orbital, which doesn't enter the above loop)
    U /= np.sqrt( global_dot(U,U*weights, comm) )  # normalize again at end (safegaurd for the zeroth orbital, which doesn't enter the above loop)
        
    return U  

# @jit()
def modifiedGramSchmidt_singleOrbital_transpose(V,weights,targetOrbital, comm):
    U = V[targetOrbital,:]
    for j in range(targetOrbital):
#         print('Orthogonalizing %i against %i' %(targetOrbital,j))
#         U -= (np.dot(V[:,targetOrbital],V[:,j]*weights) / np.dot(V[:,j],V[:,j]*weights))*V[:,j]
#         U -= np.dot(VT[targetOrbital,:],VT[j,:]*weights) *VT[j,:]
#         U /= np.sqrt( np.dot(U,U*weights) )
        
        U -= global_dot(V[targetOrbital,:],V[j,:]*weights, comm)/np.sqrt( global_dot(U,U*weights, comm) ) *V[j,:]
#         U /= np.sqrt( global_dot(U,U*weights, comm) )
    
#     U /= np.sqrt( np.dot(U,U*weights) )  # normalize again at end (safegaurd for the zeroth orbital, which doesn't enter the above loop)
    U /= np.sqrt( global_dot(U,U*weights, comm) )  # normalize again at end (safegaurd for the zeroth orbital, which doesn't enter the above loop)
        
    return U  



def modifiedGramSchmidt_singleOrbital_python(V,weights,targetOrbital):
    U = V[:,targetOrbital]
    for j in range(targetOrbital):
#         print('Orthogonalizing %i against %i' %(targetOrbital,j))
#         U -= (np.dot(V[:,targetOrbital],V[:,j]*weights) / np.dot(V[:,j],V[:,j]*weights))*V[:,j]
        U -= np.dot(V[:,targetOrbital],V[:,j]*weights) *V[:,j]
        U /= np.sqrt( np.dot(U,U*weights) )
    
    U /= np.sqrt( np.dot(U,U*weights) )  # normalize again at end (safegaurd for the zeroth orbital, which doesn't enter the above loop)
        
    return U

# @cuda.jit('void(float64[:,:], float64[:], int64, int64)')
# @njit()
def modifiedGramSchmidt_singleOrbital_C2(V,weights,targetOrbital,numPoints):
    U = V[:,targetOrbital]
    for j in range(targetOrbital):
#         print('Orthogonalizing %i against %i' %(targetOrbital,j))
        dot = 0.0
        for i in range(numPoints):
            dot += V[i,targetOrbital]*V[i,j]*weights[i]
        for i in range(numPoints):
            U[i] -= dot*V[i,j]
        norm = 0.0
        for i in range(numPoints):
            norm += U[i]*U[i]*weights[i]
        for i in range(numPoints):
            U[i] /= norm
    norm = 0.0
    for i in range(numPoints):
        norm += U[i]*U[i]*weights[i]
    for i in range(numPoints):
        U[i] /= norm
#         output[i]=U[i]
    
    return U


# @cuda.jit('void(float64[:,:], float64[:], int64, int64, float64[:])')
# def modifiedGramSchmidt_singleOrbital_GPU(V,weights,targetOrbital,numPoints, output):
#     U = V[:,targetOrbital]
#     globalID = cuda.grid(1)
#     if globalID < numPoints:
#         dot = 0.0
#         norm = 0.0
#         for j in range(targetOrbital):
#     #         print('Orthogonalizing %i against %i' %(targetOrbital,j))
#             dot += V[globalID,targetOrbital]*V[globalID,j]*weights[globalID]
# #             cuda.syncthreads()
#             U[globalID] -= dot*V[globalID,j]
# #             cuda.syncthreads()
#             norm += U[globalID]*U[globalID]*weights[globalID]
# #             cuda.syncthreads()
#             U[globalID] /= norm
# #             cuda.syncthreads()
#         norm = 0.0
# #         cuda.syncthreads()
#         norm += U[globalID]*U[globalID]*weights[globalID]
# #         cuda.syncthreads()
#         U[globalID] /= norm
# #         cuda.syncthreads()
#         output[globalID]=U[globalID]
# #         cuda.syncthreads()
#     
        

def modifiedGramSchrmidt_noNormalization(V,weights):
    n,k = np.shape(V)
    U = np.zeros_like(V)
    U[:,0] = V[:,0] 
    for i in range(1,k):
        U[:,i] = V[:,i]
        for j in range(i):
            print('Orthogonalizing %i against %i' %(i,j))
            U[:,i] -= (np.dot(U[:,i],U[:,j]*weights) / np.dot(U[:,j],U[:,j]*weights))*U[:,j]
#         U[:,i] /= np.dot(U[:,i],U[:,i]*weights)
        
    return U

def normalizeOrbitals(V,weights):
    print('Only normalizing, not orthogonalizing orbitals')
    n,k = np.shape(V)
    U = np.zeros_like(V)
#     U[:,0] = V[:,0] / np.dot(V[:,0],V[:,0]*weights)
    for i in range(0,k):
        U[:,i]  = V[:,i]
        U[:,i] /= np.sqrt( np.dot(U[:,i],U[:,i]*weights) )
        
        if abs( 1- np.dot(U[:,i],U[:,i]*weights)) > 1e-12:
            print('orbital ', i, ' not normalized? Should be 1: ', np.dot(U[:,i],U[:,i]*weights))
    
    return U



def eigenvalueNorm(psi):
    norm = np.sqrt( psi[-1]**2 )
    return norm



def clenshawCurtisNorm_withoutEigenvalue(psi):
    return np.sqrt( np.sum( psi*psi*W ) )


def s(x):
    return x*x*(3-2*x)

def m(x,tau):
    if abs(x)<=tau:
        return s(x/tau)
    elif abs(x)<=1-tau:
        return 1
    if abs(x)>1-tau:
        return s((1-x)/tau)
    


def mask(psi,x,y,z,domainSize, tau=1/16):  # a mask that damps the wavefunctions at the boundary of the domain.
    
    # domain boundaries need to be mapped to [0,1]
    mask_vector = np.ones(len(psi))
    for i in range(len(mask_vector)):
        mask_vector[i] *= m((x[i]+domainSize)/(2*domainSize),tau)
        mask_vector[i] *= m((x[i]+domainSize)/(2*domainSize),tau)
        mask_vector[i] *= m((x[i]+domainSize)/(2*domainSize),tau)
        
    
    return psi*mask_vector



def testOrthogonalization():
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size() 
    from mpiUtilities import global_dot, rprint
    
    
    N = int(1e6)
    weights=np.ones(N)
    M = 2
    targetOrbital=M-1
    V = np.random.rand(N,M)
    VT = np.transpose(V)
    out  = modifiedGramSchmidt_singleOrbital(V,weights,targetOrbital, comm)
    outT = modifiedGramSchmidt_singleOrbital_transpose(VT,weights,targetOrbital, comm)
    
    print(np.max(np.abs(out-outT)))
    
    for M in [1,2,4,8,16,32,64]:
#     for M in []:
        V = np.random.rand(N,M)
        VT = np.random.rand(M,N)
#         VT = np.copy(np.transpose(V))
        
        
        targetOrbital=M-1
        
        start=time.time()    
        out = modifiedGramSchmidt_singleOrbital(V,weights,targetOrbital, comm)
        end=time.time()
        
        startT=time.time()    
        outT = modifiedGramSchmidt_singleOrbital_transpose(VT,weights,targetOrbital, comm)
        endT=time.time()
        
#         assert np.max(np.abs(out-outT))<1e-12, "two orthogonalizations didn't agree!"
        print("Original:  Orthogonalizing %2.ith wavefunction of size %i took %f seconds" %(M,N,end-start))
        print("Transpose: Orthogonalizing %2.ith wavefunction of size %i took %f seconds\n" %(M,N,endT-startT))
    
if __name__=="__main__":
    
    testOrthogonalization()
    
    
    
#     N=200
#     domainSize=5
#     x = np.linspace(-domainSize,domainSize,N)
#     mask = np.zeros(N)
#     s_vec = np.zeros(N)
#     psi=np.ones(N)
#     
#     for i in range(N):
#         s_vec[i] = s((x[i]+domainSize)/(2*domainSize))
#         mask[i] = m((x[i]+domainSize)/(2*domainSize),1/16)
#     
#     plt.figure()
#     plt.plot(x,mask,'.',label="m(x)")
#     plt.plot(x,s_vec,'.',label="s(x)")
#     plt.show()
#     
    
    
#     masked_psi = mask(psi,x,x,x,domainSize)
#     NumWavefunctions=5
#     NumPoints=5*10**5
#     targetOrbital=1
#     n=1 # not used
#     k=1 # not used
#     
#     threadsperblock = 512
#     blockspergrid = (NumPoints + (threadsperblock - 1)) // threadsperblock
#     
#     V = np.random.rand(NumPoints,NumWavefunctions)
#     Vold = np.copy(V)
# #     np.ascontiguousarray(V)
#     weights = np.random.rand(NumPoints)
# #     np.ascontiguousarray(weights)
#     
#     # Call this once so it gets compiled.  Don't want to count that during testing
#     dummyx = modifiedGramSchmidt_singleOrbital(V[0:5,0:5],weights[0:5],targetOrbital,n,k)
#     dummyx = modifiedGramSchmidt_singleOrbital_C2(V[0:5,0:5],weights[0:5],targetOrbital,NumPoints)
#     dummyx = np.zeros(NumPoints)
#     modifiedGramSchmidt_singleOrbital_GPU[blockspergrid, threadsperblock](V,weights,targetOrbital,NumPoints, dummyx)
#     print("Done with the just-in-time compiling...")
#     
#     
#     dummy1 = np.zeros(NumPoints)
#     startTime = time.time()
# #     dummy1 = modifiedGramSchmidt_singleOrbital(V,weights,targetOrbital,n,k)
#     dummy1 = modifiedGramSchmidt_singleOrbital_C2(V,weights,targetOrbital,NumPoints)
#     endTime = time.time()
#     print("C version took ", endTime-startTime)
#     
#     startTime = time.time()
#     dummy2 = modifiedGramSchmidt_singleOrbital_python(V,weights,targetOrbital)
#     endTime = time.time()
#     print("Python version took ", endTime-startTime)
#     
#     
#     dummy3=np.zeros(NumPoints)
#     startTime = time.time()
#     modifiedGramSchmidt_singleOrbital_GPU[blockspergrid, threadsperblock](V,weights,targetOrbital,NumPoints, dummy3)
#     endTime = time.time()
#     print("CUDA version took ", endTime-startTime)
#     
#     print("Same result?: ",np.allclose(dummy1,dummy2))
#     print("Same result?: ",np.allclose(dummy1,dummy3))
#     if not ( (np.allclose(dummy1,dummy2)) or (np.allclose(dummy1,dummy3)) ):
#         print(dummy1[:5])
#         print(dummy2[:5])
#         print(dummy3[:5])
#         
#     if not np.allclose(V,Vold):
#         print("V not close to Vold")
#         if np.allclose(V[:,targetOrbital], dummy3):
#             print("But V[:,targetOrbital] contains the orth wave.")
#         for i in range(NumWavefunctions):
#             if i != targetOrbital:
#                 if not np.allclose(V[:,i], Vold[:,i]):
#                     print("Column %i  also differ", i)
#     # Interesting.  My python version is slightly faster than my C version.  Python is using all numpy functions, so that's why.
#     # But my non-numpy version, compiled with numba, is faster! 
#     
    
    
    