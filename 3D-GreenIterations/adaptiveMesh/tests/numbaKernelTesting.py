import numpy as np
from numba import cuda, float64
from timeit import default_timer as timer


        
@cuda.jit
def interact(targets,sources,interactions):
    globalID = cuda.grid(1)
    xt, yt, zt = targets[globalID]
    for i in range(len(sources)):
        xs, ys, zs = sources[i]
        if not ( (xs==xt) and (ys==yt) and (zs==zt) ):
            interactions[globalID] += -1 / ( (xt-xs)**2 + (yt-ys)**2 + (zt-zs)**2 )**0.5
            
@cuda.jit
def interactShared(targets,sources,interactions):
    globalID = cuda.grid(1)
    localID = cuda.threadIdx.x
    xt, yt, zt = targets[globalID]
    sharedSourceX = cuda.shared.array(shape=(threadsPerBlock), dtype=float64)
    sharedSourceY = cuda.shared.array(shape=(threadsPerBlock), dtype=float64)
    sharedSourceZ = cuda.shared.array(shape=(threadsPerBlock), dtype=float64)
    

    for i in range(len(sources)/blocksPerGrid):
        sharedSourceX[localID] = sources[i*threadsPerBlock + localID][0]
        sharedSourceY[localID] = sources[i*threadsPerBlock + localID][1]
        sharedSourceZ[localID] = sources[i*threadsPerBlock + localID][2]
    
        cuda.syncthreads()
    
        for j in range(threadsPerBlock):
            xs = sharedSourceX[j]
            ys = sharedSourceY[j]
            zs = sharedSourceZ[j]
            if not ( (xs==xt) and (ys==yt) and (zs==zt) ):
                interactions[globalID] += -1 / ( (xt-xs)**2 + (yt-ys)**2 + (zt-zs)**2 )**0.5
        
        cuda.syncthreads()
      
  
                
def serialInteract(targets,sources,interactions):
    for i in range(len(targets)):
        xt,yt,zt = targets[i]
        for j in range(len(sources)):
            xs,ys,zs = sources[j]
            if not ( (xs==xt) and (ys==yt) and (zs==zt) ):
                interactions[i] += -1 / ( (xt-xs)**2 + (yt-ys)**2 + (zt-zs)**2 )**0.5
     
        

N = 2**10
threadsPerBlock = 256
blocksPerGrid = (N + (threadsPerBlock - 1)) // threadsPerBlock
print('Number of particles:  ', N)
print('Threads per block:    ', threadsPerBlock)
print('Blocks per grid:      ', blocksPerGrid)

targets = np.random.rand(N,3)
sources = np.random.rand(N,3)


interactions = np.zeros(N)
startTime = timer()
interact[blocksPerGrid, threadsPerBlock](targets,sources,interactions)
cudaKernelTime = timer() - startTime
print(interactions[:3]/N,' ... ', interactions[-3:]/N)
hostInteractions = interactions

interactions = np.zeros(N)
startTime = timer()
interactShared[blocksPerGrid, threadsPerBlock](targets,sources,interactions)
cudaSharedMemKernelTime = timer() - startTime
print(interactions[:3]/N,' ... ', interactions[-3:]/N)
hostSharedInteractions = interactions
print('Max difference: ', np.max( hostInteractions-hostSharedInteractions))


# interactions = np.zeros(N)
# startTime = timer()
# serialInteract(targets,sources,interactions)
# serialTime = timer() - startTime
# print(interactions[:3]/N,' ... ', interactions[-3:]/N)

print('CUDA Kernel took:        %f seconds.' %cudaKernelTime)
print('CUDA Shared Mem took:    %f seconds.' %cudaSharedMemKernelTime)
# print('Serial loop took:    %f seconds.' %serialTime)