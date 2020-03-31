import numpy as np
import ctypes
from mpi4py import MPI
import sys
import time
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

from mpiUtilities import rprint
import orthogonalization_wrapper as orth

if __name__=="__main__":
    numPoints=int(sys.argv[1])
    numWavefunctions=int(sys.argv[2])
    
    wavefunctions=np.random.rand(numWavefunctions,numPoints)
    W = np.ones(numPoints)
    # print(wavefunctions)
    print(np.shape(wavefunctions))
    gpuPresent=False
    
    
    # for j in range(numPoints):
    #     print(wavefunctions[:,j])
    
#     for j in range(numPoints):
#         for i in range(numWavefunctions):
#             wavefunctions[i,j] = (j+1)**(i+1) - (5-j)*(rank)**(2);
        
        
    # targetWavefunction=3
    start=time.time()
    for targetWavefunction in range(numWavefunctions):
        U=np.copy(wavefunctions[targetWavefunction])
        wavefunctions = orth.callOrthogonalization(wavefunctions, U, W, targetWavefunction, gpuPresent)
    end=time.time()
    
#     for j in range(numPoints):
    print(wavefunctions[:,0])
       
    # check orthogonalization 
    for i in range(numWavefunctions):
        for j in range(i):
            overlap=abs(np.dot(wavefunctions[i,:],wavefunctions[j,:]))
            if overlap/numPoints>1e-12:
                print("overlap between %i and %i = %1.3e." %(i,j,overlap))
    
    
    # check normalization
    for i in range(numWavefunctions):
        norm=abs(np.dot(wavefunctions[i,:],wavefunctions[i,:]))
        if (norm-1)/numPoints>1e-12:
            print("norm-1 of wavefunction %i = %1.3e." %(i,norm-1))
      
    print("Python time to orthogonalize %i wavefnctions of %i points distributed over %i precessors: %f seconds" %(numWavefunctions,numPoints,size,end-start))  
    
#     start2=time.time()x
    from orthogonalizationRoutines import testOrthogonalization
    testOrthogonalization(numPoints,numWavefunctions)        
#     end2=time.time()
#     print("Python time to orthogonalize %i wavefnctions of %i points distributed over %i precessors: %f seconds" %(numWavefunctions,numPoints,size,end-start))  
    
    
    