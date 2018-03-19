'''
Created on Mar 14, 2018
 
@author: nathanvaughn
'''
import numpy as np
import numba
from numba import cuda, vectorize, float64
# print(numba.__version__)
# print('Hello from vector add')
# print('updated')
# print(numba.jit.__doc__)
from timeit import default_timer as timer
# from numbapro import vectorize
 
def serialAdd(A,B,C):
    for i in range(A.size):
        C[i] = A[i] + B[i]
  
@numba.jit("f8(f8[:],f8[:])")
def vectorAdd(a,b):
    for i in range(100):
        a += 1
    return a+b

@vectorize(["float64(float64,float64)"], target='cuda')
def cudaAdd(a,b):
    for i in range(100):
        a += 1
    return a+b

  
#   
# A = np.ones(50000000)
# B = 2*np.ones(50000000)
# C = np.zeros(A.size)
# # startTime = timer()
# # serialAdd(A,B,C)
# # serialTime = timer() - startTime
# #   
# # print(C[:5])
# # print(C[-5:])
# # print('Serial Add took %f seconds.' %serialTime)
#   
# startTime = timer()
# C = vectorAdd(A,B)
# vectorizedTime = timer() - startTime
#    
# print(C[:5])
# print(C[-5:])
# print('Vectorized Add took %f seconds.' %vectorizedTime)
#   
#   
# C = np.zeros(A.size)
# startTime = timer()
#   
# # cudaAdd[griddim, blockdim](aryA, aryB)
# C = cudaAdd(A,B)
# cudaTime = timer() - startTime
#     
# print(C[:5])
# print(C[-5:])
# print('CUDA Add took %f seconds.' %cudaTime)


@vectorize(["float64(float64,float64,float64,float64,float64,float64)"], target='cuda')
def cudaVecotrizedInteraction(xt,yt,zt,xs,ys,zs):
    rsq =  (xt-xs)**2 + (yt-ys)**2 + (zt-zs)**2 
    return rsq

# def cudaKernelInteraction(xt,yt,zt,xs,ys,zs):
#     rsq =  (xt-xs)**2 + (yt-ys)**2 + (zt-zs)**2 
#     return rsq

def serialInteract(XT,YT,ZT,XS,YS,ZS):
    Rsq = np.zeros(XT.size)
    for i in range(XT.size):
        for j in range(XS.size):
            Rsq[i] += (XT[i]-XS[j])**2 + (YT[i]-YS[j])**2 + (ZT[i]-ZS[j])**2 
    return Rsq

N = 5000
XT = np.random.rand(N)
YT = np.random.rand(N)
ZT = np.random.rand(N)
XS = np.random.rand(N)
YS = np.random.rand(N)
ZS = np.random.rand(N)

# startTime = timer()
# R1 = serialInteract(XT,YT,ZT,XS,YS,ZS)
# print('R1 shape: ', R1.size)
# serialTime = timer() - startTime
# print(R1[:5])

R2 = np.zeros(N)
startTime = timer()
for i in range(N):
    R2[i] += np.sum( np.array( cudaVecotrizedInteraction(XT[i],YT[i],ZT[i],XS,YS,ZS) ) )
#     R2 +=  cudaInteraction(XT,YT,ZT,XS[i],YS[i],ZS[i]) 
print('R2 shape: ', R2.size)
cudaTime = timer() - startTime
print(R2[:5])

# print('Serial Interaction took:  %f seconds.' %serialTime)
print('CUDA Interaction took:    %f seconds.' %cudaTime)