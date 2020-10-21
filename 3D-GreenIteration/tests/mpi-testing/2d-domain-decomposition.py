from mpi4py import MPI
import numpy as np
import time
import numba
from numpy import float32, float64, int32
import array
import pickle


def discretizeRegion(xlow,xhigh,nx,ylow,yhigh,ny):
    x = np.linspace(xlow,xhigh,nx)
    y = np.linspace(ylow,yhigh,ny)
    X,Y = np.meshgrid(x,y)    
    return X,Y

@numba.njit()
def interact(targetX,targetY,sourceX,sourceY,potential):
    # Loop over targets
    for i in range(targetX.shape[0]):
        for j in range(targetX.shape[1]):
            
            # Loop over sources
            for ii in range(sourceX.shape[0]):
                for jj in range(sourceX.shape[1]):
                    
                    
                    r = np.sqrt( (targetX[i,j]-sourceX[ii,jj])**2 + (targetY[i,j]-sourceY[ii,jj])**2 )
                    if r > 1e-16:  
                        potential[i,j] += 1/r
            
    return
    

def testInteraction():
    
    X,Y = discretizeRegion(0,1,2,0,1,2)
    potential = np.zeros_like(X)
    interact(X,Y,X,Y,potential)
    try:
        assert np.allclose(potential, np.array([[2.70710678, 2.70710678],[2.70710678, 2.70710678]]))
        print('2x2 Test Passed.')
    except AssertionError as e:
        print('2x2 Test Failed: ')


def mpiRun(xlow,xhigh,nx,ylow,yhigh,ny):
    

    
    
    NORTH = 0
    SOUTH = 1
    EAST = 2
    WEST = 3
    
    comm = MPI.COMM_WORLD
    mpi_rows = int(np.floor(np.sqrt(comm.size)))
    mpi_cols = comm.size // mpi_rows
    if mpi_rows*mpi_cols > comm.size:
        mpi_cols -= 1
    if mpi_rows*mpi_cols > comm.size:
        mpi_rows -= 1
        
        
    rank = comm.Get_rank()
    if (rank==0): print("Creating a %d x %d processor grid..." % (mpi_rows, mpi_cols) )
    ccomm = comm.Create_cart(dims = (mpi_rows, mpi_cols),periods =(False,False),reorder=False)
    coord2d = ccomm.Get_coords(rank)
    print ("In 2D topology, Processor ",rank, " has coordinates ",coord2d)
    
    my_mpi_row, my_mpi_col = ccomm.Get_coords( ccomm.rank ) 
    neigh = [0,0,0,0]
    
    neigh[NORTH], neigh[SOUTH] = ccomm.Shift(direction=0, disp=1)
    neigh[EAST],  neigh[WEST]  = ccomm.Shift(direction=1, disp=1)
    

    if rank==0:
        startTime = MPI.Wtime()

    x_per_process = nx // mpi_rows
    y_per_process = ny // mpi_cols
    x_global = np.linspace(xlow,xhigh,nx)
    y_global = np.linspace(ylow,yhigh,ny)
    
    x_start = coord2d[0]*x_per_process
    x_end   = x_start + x_per_process
    
    y_start = coord2d[1]*y_per_process
    y_end   = y_start + y_per_process
    
    X, Y = np.meshgrid( x_global[x_start:x_end], y_global[y_start:y_end])
    XS = np.zeros_like(X)
    YS = np.zeros_like(Y)
    pickleSize = len(pickle.dumps(X,-1))
#     print("Array size: ", X.nbytes)
#     print('PickleSize = ', pickleSize)
    buffer1 = bytearray(pickleSize+200)  # The pickle is slightly larger than the array itself.  Buffer needs to fit the pickle.
    buffer2 = bytearray(pickleSize+200)  # The pickle is slightly larger than the array itself.  Buffer needs to fit the pickle.
    
    potential = np.zeros_like(X)
    singleInteractStart = time.time()
    interact(X,Y,X,Y,potential)
    singleInteractStop = time.time()
    print('Time for self-interaction: ', singleInteractStop-singleInteractStart)
    
    ## Compute interactions with other processors
    for i in range(1, comm.size):
#         print("i = ",i)
#         ccomm.barrier()
        
        sendTo = (rank+i) % comm.size
        recieveFrom = (rank-i) % comm.size
#         print("Processor %i sending to %i, receiving from %i." %(rank,sendTo, recieveFrom) )

        commStart = time.time()
        req0 = ccomm.isend(X,sendTo, tag=2**i+0)
        req1 = ccomm.isend(Y,sendTo, tag=2**i+1)
#         print('processors %i finished sending.' %rank)
#         ccomm.barrier()
        req2 = ccomm.irecv(source=recieveFrom, buf=buffer1, tag=2**i+0)
        req3 = ccomm.irecv(source=recieveFrom, buf=buffer2, tag=2**i+1)
        
        
        XS = req2.wait()
        YS = req3.wait()
        req1.wait()
        req2.wait()
        commStop = time.time()
        print("Processor %i sending to %i, receiving from %i.  Took %f seconds" %(rank,sendTo, recieveFrom, commStop-commStart))
#         ccomm.barrier()

        
        interact(X,Y,XS,YS,potential)
    
    local_sum = np.sum(potential)
#     print(np.sum(potential) )
    print("Processor %i local sum: %f" %(rank,local_sum))
    
    global_sum = ccomm.reduce(local_sum)
    if rank==0:
        print('Global sum: ', global_sum)
        
    if rank==0:
        endTime = MPI.Wtime()
        totalTime = endTime-startTime
        print('Total time: ', totalTime)
    

    

def test_MPI_version():
    mpiRun(0,2,1,2,1)


if __name__=="__main__":    
#     testInteraction()
    
    mpiRun(0,1,180,0,1,180)
    
    MPI.Finalize