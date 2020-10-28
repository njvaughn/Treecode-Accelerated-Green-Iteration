import numpy as np
import ctypes
from mpi4py import MPI
import sys
import time
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

from mpiUtilities import rprint, global_dot
import orthogonalization_wrapper as orth
import moveData_wrapper as MD

if __name__=="__main__":
    numPoints=int(sys.argv[1])
    numWavefunctions=int(sys.argv[2])
    
    if int(sys.argv[3])==0:
        gpuPresent=False
    elif int(sys.argv[3])==1:
        gpuPresent=True
    else:
        print("What was sys.argv[3] supposed to be?  It was ", sys.argv[3])
        exit(-1)
    
    print("rank = ", rank)
    np.random.seed(rank)
    wavefunctions=np.random.rand(numWavefunctions,numPoints)
    
    wavefunctions=np.zeros( (numWavefunctions,numPoints), order='C')
    W = np.ones(numPoints)
#     gpuPresent=True 
    
 
    wavefunctions=np.random.rand(numWavefunctions,numPoints)
    

    X=np.ones(10) #dummy array to init GPU
    if gpuPresent: MD.callCopyVectorToDevice(X)   
    
#     print("Before copyin: ", wavefunctions[:,0])
      
    if gpuPresent: MD.callCopyVectorToDevice(W)
    if gpuPresent: MD.callCopyVectorToDevice(wavefunctions)

#     U=np.copy(wavefunctions[0])
#     MD.callCopyVectorToDevice(U)
    start=time.time()
    for targetWavefunction in range(numWavefunctions):
        U=np.copy(wavefunctions[targetWavefunction])
        if gpuPresent: MD.callCopyVectorToDevice(U)
        orth.callOrthogonalization(wavefunctions, U, W, targetWavefunction, gpuPresent)
        if gpuPresent: MD.callRemoveVectorFromDevice(U)
    
    
    end=time.time()
    if gpuPresent: MD.callCopyVectorFromDevice(wavefunctions)
    
#     print("After copyout: ", wavefunctions[:,0])
#     input()
       
    check=True
    # check orthogonalization
    if check==True:
        flag=0 
        for i in range(numWavefunctions):
            for j in range(i):
                overlap=abs(global_dot(wavefunctions[i,:],wavefunctions[j,:],comm))
                if overlap/numPoints>1e-12:
                    flag=1
    #                 print("overlap between %i and %i = %1.3e." %(i,j,overlap))
        if flag==1:
            print("Error in overlaps.")
        
        
        # check normalization
        flag=0
        for i in range(numWavefunctions):
            norm=abs(global_dot(wavefunctions[i,:],wavefunctions[i,:],comm))
            if (norm-1)/numPoints>1e-12:
                flag=1
    #             print("norm-1 of wavefunction %i = %1.3e." %(i,norm-1))
        if flag==1:
            print("Error in norms.")
      
    print("Python time to orthogonalize %i wavefunctions of %i points distributed over %i precessors: %f seconds" %(numWavefunctions,numPoints,size,end-start))  
    
    
    
# #     start2=time.time()x
#     from orthogonalizationRoutines import testOrthogonalization
#     testOrthogonalization(numPoints,numWavefunctions)        
# #     end2=time.time()
#     print("Python time to orthogonalize %i wavefnctions of %i points distributed over %i precessors: %f seconds" %(numWavefunctions,numPoints,size,end-start))  
    
    
    