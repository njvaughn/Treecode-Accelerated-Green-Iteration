import numpy as np
import ctypes
from mpi4py import MPI
import gc

from mpiUtilities import rprint


try: 
    zoltanLoadBalancing = ctypes.CDLL('ZoltanRCB.so')
except OSError as e:
    print(e)
    print("Exiting since ZoltanRCB.so could not be loaded.")
    exit(-1)

        

        
""" Set argtypes of the wrappers. """

try:
    zoltanLoadBalancing.loadBalanceRCB.argtypes = (
                                                    ctypes.POINTER(ctypes.POINTER(ctypes.c_double)), 
                                                    ctypes.POINTER(ctypes.POINTER(ctypes.c_double)), 
                                                    ctypes.POINTER(ctypes.POINTER(ctypes.c_double)), 
                                                    ctypes.POINTER(ctypes.POINTER(ctypes.c_double)), 
                                                    ctypes.POINTER(ctypes.POINTER(ctypes.c_double)), 
                                                    ctypes.POINTER(ctypes.POINTER(ctypes.c_double)), 
                                                    ctypes.POINTER(ctypes.POINTER(ctypes.c_int)), 
                                                    ctypes.POINTER(ctypes.POINTER(ctypes.c_int)), 
                                                    ctypes.c_int, 
                                                    ctypes.c_int, 
                                                    ctypes.POINTER(ctypes.c_int) 
                                                    )
except NameError:
    print("Could not set argtypes of zoltanLoadBalancing")



def callZoltan(cellsX, cellsY, cellsZ, cellsDX, cellsDY, cellsDZ, coarsePtsPerCell, finePtsPerCell, numCells, globalStart):
    
    numCells=len(cellsX)
    newNumCells=np.array(0,dtype=np.int)
    
    c_double_p = ctypes.POINTER(ctypes.c_double)    # standard pointer to array of doubles
    c_int_p = ctypes.POINTER(ctypes.c_int)    # standard pointer to array of doubles
    

    cellsX_p = cellsX.ctypes.data_as(c_double_p)
    cellsY_p = cellsY.ctypes.data_as(c_double_p)
    cellsZ_p = cellsZ.ctypes.data_as(c_double_p)
    cellsDX_p = cellsDX.ctypes.data_as(c_double_p)
    cellsDY_p = cellsDY.ctypes.data_as(c_double_p)
    cellsDZ_p = cellsDZ.ctypes.data_as(c_double_p)

    coarsePtsPerCell_p = coarsePtsPerCell.ctypes.data_as(c_int_p)
    finePtsPerCell_p = finePtsPerCell.ctypes.data_as(c_int_p)


    newNumCells_p = newNumCells.ctypes.data_as(c_int_p)
    
    
#     print("Rank %i, Before call: cellsX = " %rank, cellsX[:5])
    zoltanLoadBalancing.loadBalanceRCB(
                                        ctypes.byref(cellsX_p), 
                                        ctypes.byref(cellsY_p),
                                        ctypes.byref(cellsZ_p),
                                        ctypes.byref(cellsDX_p),
                                        ctypes.byref(cellsDY_p),
                                        ctypes.byref(cellsDZ_p),
                                        ctypes.byref(coarsePtsPerCell_p),
                                        ctypes.byref(finePtsPerCell_p),
                                        ctypes.c_int(numCells),
                                        ctypes.c_int(globalStart),
                                        newNumCells_p
                                        )
    
#     print("Rank %i, After call: cellsX = " %rank, cellsX_p[:5])
    print("Calling garbage collector")
    gc.collect()
    print("garbage collection complete.")
    return cellsX_p[:newNumCells], cellsY_p[:newNumCells], cellsZ_p[:newNumCells], cellsDX_p[:newNumCells], cellsDY_p[:newNumCells], cellsDZ_p[:newNumCells], coarsePtsPerCell_p[:newNumCells], finePtsPerCell_p[:newNumCells], newNumCells


if __name__=="__main__":
    
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    r=15
    numCells=r*(rank+1)
    
    np.random.seed(rank)
    cellsX=2*np.random.rand(numCells)-1
    cellsY=2*np.random.rand(numCells)-1
    cellsZ=2*np.random.rand(numCells)-1
    cellsDX=np.ones(numCells)
    cellsDY=np.ones(numCells)
    cellsDZ=np.ones(numCells)
    
    coarsePtsPerCell=(rank+1)*np.ones(numCells, dtype=np.int32)
    finePtsPerCell=-(rank+1)*np.ones(numCells, dtype=np.int32)
    print("rank %i before balancing, coarse points per cell: " %rank, coarsePtsPerCell)
    print("rank %i before balancing, fine points per cell:   " %rank, finePtsPerCell)

    print(type(coarsePtsPerCell))
    print(coarsePtsPerCell.dtype)
    
    globalStart=0
    
    for i in range(rank):
        globalStart += r*(i+1)
    print("rank %i starts at %i and has %i cells. " %(rank,globalStart,numCells))
    
    newCellsX, newCellsY, newCellsZ, newCellsDX, newCellsDY, newCellsDZ, newCoarsePtsPerCell, newFinePtsPerCell, newNumCells = callZoltan(cellsX, cellsY, cellsZ, cellsDX, cellsDY, cellsDZ, coarsePtsPerCell, finePtsPerCell, numCells, globalStart)
    
    xmean=np.mean(newCellsX)
    ymean=np.mean(newCellsY)
    zmean=np.mean(newCellsZ)
      
    rprint(0,"len(newCellsX) = ", len(newCellsX)) 
    rprint(0,"newNumCells = ", newNumCells)    
    assert len(newCellsX)==newNumCells, "Length of cellsX does not equal newNumCells."  
    print("After load balancing, rank %i has %i cells cenetered at (x,y,z)=(%f,%f,%f)." %(rank,len(newCellsX),xmean,ymean,zmean))

    comm.Barrier()    
    print("Plot the points from each rank now.")
    
    
    comm.Barrier()
    print("rank %i, coarse points per cell: " %rank, newCoarsePtsPerCell)
    comm.Barrier()
    print("rank %i, fine points per cell: " %rank, newFinePtsPerCell)
    
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     color=np.random.rand(3,)
# #     for i in range(len(newCellsX)):
# #         ax.scatter(newCellsX[i],newCellsY[i],newCellsZ[i],'o',c=[newCoarsePtsPerCell[i]*size/255],label="rank %i" %rank)
#     ax.scatter(newCellsX,newCellsY,newCellsZ,'o',color=color,label="rank %i" %rank)
#     ax.set_xlim([-1,1])
#     ax.set_ylim([-1,1])
#     ax.set_zlim([-1,1])
#     plt.title("Rank %i" %rank)
#     comm.Barrier()
#     plt.show()
    



