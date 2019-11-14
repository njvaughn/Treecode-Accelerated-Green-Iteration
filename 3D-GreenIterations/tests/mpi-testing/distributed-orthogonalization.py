from mpi4py import MPI
import numpy as np
import time
import numba
from numpy import float32, float64, int32
import array
import pickle


def global_dot(u,v,comm):
    local_dot = np.dot(u,v)
    global_dot = comm.allreduce(local_dot)
    return global_dot
    
    


def mpiRun(numWavefunctions,numPoints):
    

    
    comm = MPI.COMM_WORLD
    
    numProcs = comm.size
    rank = comm.Get_rank()
    
#     if rank==0:
#         startTime = MPI.Wtime()
    
#     print('numProcs = ', numProcs)
    
#     localArrayLength = numPoints//numProcs
    localArrayLength = -(-numPoints // numProcs)
    
    if rank==0:
        localArrayLength += (numPoints - numProcs*localArrayLength) 
    print('Rank %i has %i points.' %(rank,localArrayLength) )
        
    global_ArrayLength = comm.reduce(localArrayLength)
    if rank==0: assert global_ArrayLength==numPoints, 'Sum of local array lengths not equal to numPoints'
    
    xlow = rank/numProcs
    xhigh = (rank+1)/numProcs
    xvec = np.linspace(xlow,xhigh,localArrayLength)
    
    wavefunctions = np.zeros( (numWavefunctions,localArrayLength) )
    for i in range(numWavefunctions):
        wavefunctions[i,:] = np.sin((i+1)*xvec)
    
    
    if rank==0:
        startTime = MPI.Wtime()
        
    for i in range(numWavefunctions):
        for j in range(i):
            dot = global_dot(wavefunctions[i,:], wavefunctions[j,:],comm)
            if rank==0: print(dot)
 
    
    if rank==0:
        endTime = MPI.Wtime()
        totalTime = endTime-startTime
        print('Total time: ', totalTime)
    


if __name__=="__main__":    
    
    mpiRun(5,10000000)
    
    MPI.Finalize