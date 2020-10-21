'''
Wrapper for calling the zoltan load balancing.
Alternative is to use pyZoltan, although that gave me a lot of trouble.
'''
import numpy as np
import ctypes
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
from mpiUtilities import rprint
import gc



try: 
    zoltanLoadBalancing = ctypes.CDLL('ZoltanRCB.so')
except OSError as e:
    rprint(rank,e)
    rprint(rank,"Exiting since ZoltanRCB.so could not be loaded.")
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
    rprint(rank,"Could not set argtypes of zoltanLoadBalancing")



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
    
    
#     rprint(0,"Rank %i, Before call: cellsX = " %rank, cellsX[:5])
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
    
#     rprint(0,"Rank %i, After call: cellsX = " %rank, cellsX_p[:5])
#     rprint(rank,"Calling garbage collector")
#     gc.collect()
#     rprint(rank,"garbage collection complete.")

#     rprint(rank,"numCells            = ", numCells)
#     rprint(rank,"newNumCells         = ", newNumCells)
#     rprint(rank, "length of cellsX_p = ", len(cellsX_p[:newNumCells]))
#     
#     cellsX = np.copy(cellsX_p[:newNumCells])
#     del cellsX_p
#     
#     cellsY = np.copy(cellsY_p[:newNumCells])
#     del cellsY_p
#     
#     cellsZ = np.copy(cellsZ_p[:newNumCells])
#     del cellsZ_p
#     
#     cellsDX = np.copy(cellsDX_p[:newNumCells])
#     del cellsDX_p
#     
#     cellsDY = np.copy(cellsDY_p[:newNumCells])
#     del cellsDY_p
#     
#     cellsDZ = np.copy(cellsDZ_p[:newNumCells])
#     del cellsDZ_p
#     
#     coarsePtsPerCell = np.copy(coarsePtsPerCell_p[:newNumCells])
#     del coarsePtsPerCell_p
#     
#     finePtsPerCell = np.copy(finePtsPerCell_p[:newNumCells])
#     del finePtsPerCell_p
    
    return cellsX_p[:newNumCells], cellsY_p[:newNumCells], cellsZ_p[:newNumCells], cellsDX_p[:newNumCells], cellsDY_p[:newNumCells], cellsDZ_p[:newNumCells], coarsePtsPerCell_p[:newNumCells], finePtsPerCell_p[:newNumCells], newNumCells
#     return cellsX, cellsY, cellsZ, cellsDX, cellsDY, cellsDZ, coarsePtsPerCell, finePtsPerCell, newNumCells

def dummyFunction(cellsX, cellsY, cellsZ, cellsDX, cellsDY, cellsDZ, coarsePtsPerCell, finePtsPerCell, numCells, globalStart):
    newCellsX, newCellsY, newCellsZ, newCellsDX, newCellsDY, newCellsDZ, newCoarsePtsPerCell, newFinePtsPerCell, newNumCells = callZoltan(cellsX, cellsY, cellsZ, cellsDX, cellsDY, cellsDZ, coarsePtsPerCell, finePtsPerCell, numCells, globalStart)
    return newCellsX, newCellsY, newCellsZ, newCellsDX, newCellsDY, newCellsDZ, newCoarsePtsPerCell, newFinePtsPerCell, newNumCells
    
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
    rprint(0,"rank %i before balancing, coarse points per cell: " %rank, coarsePtsPerCell)
    rprint(0,"rank %i before balancing, fine points per cell:   " %rank, finePtsPerCell)
 
#     rprint(0,type(coarsePtsPerCell))
#     rprint(0,coarsePtsPerCell.dtype)
     
    globalStart=0
     
    for i in range(rank):
        globalStart += r*(i+1)**3
    rprint(0,"rank %i starts at %i and has %i cells. " %(rank,globalStart,numCells))
     
#     newCellsX, newCellsY, newCellsZ, newCellsDX, newCellsDY, newCellsDZ, newCoarsePtsPerCell, newFinePtsPerCell, newNumCells = dummyFunction(cellsX, cellsY, cellsZ, cellsDX, cellsDY, cellsDZ, coarsePtsPerCell, finePtsPerCell, numCells, globalStart)
    newCellsX, newCellsY, newCellsZ, newCellsDX, newCellsDY, newCellsDZ, newCoarsePtsPerCell, newFinePtsPerCell, newNumCells = callZoltan(cellsX, cellsY, cellsZ, cellsDX, cellsDY, cellsDZ, coarsePtsPerCell, finePtsPerCell, numCells, globalStart)
     
    del cellsX
    del cellsY
    del cellsZ
    del cellsDX
    del cellsDY
    del cellsDZ   
    del coarsePtsPerCell
    del finePtsPerCell
    del numCells
    del globalStart
    rprint(rank,"calling garbage collector.")
    comm.barrier()
    gc.collect()
    comm.barrier()
    
    xmean=np.mean(newCellsX)
    ymean=np.mean(newCellsY)
    zmean=np.mean(newCellsZ)
       
    rprint(0,"len(newCellsX) = ", len(newCellsX)) 
    rprint(0,"newNumCells = ", newNumCells)    
    assert len(newCellsX)==newNumCells, "Length of cellsX does not equal newNumCells."  
    rprint(0,"After load balancing, rank %i has %i cells cenetered at (x,y,z)=(%f,%f,%f)." %(rank,len(newCellsX),xmean,ymean,zmean))
 
    comm.Barrier()    
    rprint(0,"Plot the points from each rank now.")
     
     
    comm.Barrier()
    rprint(0,"rank %i, coarse points per cell: " %rank, newCoarsePtsPerCell)
    comm.Barrier()
    rprint(0,"rank %i, fine points per cell: " %rank, newFinePtsPerCell)
    
    rprint(rank,"calling garbage collector.")
    comm.barrier()
    gc.collect()
    comm.barrier()

    
    
     
# #     fig = plt.figure()
# #     ax = fig.add_subplot(111, projection='3d')
# #     color=np.random.rand(3,)
# # #     for i in range(len(newCellsX)):
# # #         ax.scatter(newCellsX[i],newCellsY[i],newCellsZ[i],'o',c=[newCoarsePtsPerCell[i]*size/255],label="rank %i" %rank)
# #     ax.scatter(newCellsX,newCellsY,newCellsZ,'o',color=color,label="rank %i" %rank)
# #     ax.set_xlim([-1,1])
# #     ax.set_ylim([-1,1])
# #     ax.set_zlim([-1,1])
# #     plt.title("Rank %i" %rank)
# #     comm.Barrier()
# #     plt.show()
#     
# 


