import numpy as np
import ctypes
from mpi4py import MPI

from mpiUtilities import rprint


try: 
    zoltanLoadBalancing = ctypes.CDLL('ZoltanRCB.so')
except OSError as e:
    print(e)
    exit(-1)

        

        
""" Set argtypes of the wrappers. """

try:
#     _cpu_orthogonalizationRoutines.modifiedGramSchmidt_singleWavefunction.argtypes = (np.ctypeslib.ndpointer(dtype=np.intp), 
    zoltanLoadBalancing.loadBalanceRCB.argtypes = (
                                                    ctypes.POINTER(ctypes.POINTER(ctypes.c_double)), 
                                                    ctypes.POINTER(ctypes.POINTER(ctypes.c_double)), 
                                                    ctypes.POINTER(ctypes.POINTER(ctypes.c_double)), 
                                                    ctypes.POINTER(ctypes.POINTER(ctypes.c_double)), 
                                                    ctypes.POINTER(ctypes.POINTER(ctypes.c_double)), 
                                                    ctypes.POINTER(ctypes.POINTER(ctypes.c_double)), 
                                                    ctypes.c_int, 
                                                    ctypes.c_int, 
                                                    ctypes.POINTER(ctypes.c_int) 
                                                    )
except NameError:
    print("Could not set argtypes of zoltanLoadBalancing")



def callZoltan(cellsX, cellsY, cellsZ, cellsDX, cellsDY, cellsDZ, numCells, globalStart):
    
    numCells=len(cellsX)
    newNumCells=np.array(0,dtype=np.int)

#     numWavefunctions,numPoints = np.shape(wavefunctions)
    
#     print("numWavefunctions,numPoints = ", numWavefunctions,numPoints)
    c_double_p = ctypes.POINTER(ctypes.c_double)    # standard pointer to array of doubles
    c_int_p = ctypes.POINTER(ctypes.c_int)    # standard pointer to array of doubles
    c_double_pp = ctypes.POINTER(ctypes.POINTER(ctypes.c_double))  # double pointer (for 2D arrays of doubles)
    
    
#     cellsX_pp = cellsX.ctypes.data_as(c_double_pp)
#     cellsY_pp = cellsY.ctypes.data_as(c_double_pp)
#     cellsZ_pp = cellsZ.ctypes.data_as(c_double_pp)
#     cellsDX_pp = cellsDX.ctypes.data_as(c_double_pp)
#     cellsDY_pp = cellsDY.ctypes.data_as(c_double_pp)
#     cellsDZ_pp = cellsDZ.ctypes.data_as(c_double_pp)
    
    cellsX_p = cellsX.ctypes.data_as(c_double_p)
    cellsY_p = cellsY.ctypes.data_as(c_double_p)
    cellsZ_p = cellsZ.ctypes.data_as(c_double_p)
    cellsDX_p = cellsDX.ctypes.data_as(c_double_p)
    cellsDY_p = cellsDY.ctypes.data_as(c_double_p)
    cellsDZ_p = cellsDZ.ctypes.data_as(c_double_p)


    newNumCells_p = newNumCells.ctypes.data_as(c_int_p)
    
    
    print("Rank %i, Before call: cellsX = " %rank, cellsX[:5])
    zoltanLoadBalancing.loadBalanceRCB(
                                        ctypes.byref(cellsX_p), 
                                        ctypes.byref(cellsY_p),
                                        ctypes.byref(cellsZ_p),
                                        ctypes.byref(cellsDX_p),
                                        ctypes.byref(cellsDY_p),
                                        ctypes.byref(cellsDZ_p),
                                        ctypes.c_int(numCells),
                                        ctypes.c_int(globalStart),
                                        newNumCells_p
                                        )
    
#     zoltanLoadBalancing.loadBalanceRCB(
#                                         cellsX_pp, 
#                                         cellsY_pp,
#                                         cellsZ_pp,
#                                         cellsDX_pp,
#                                         cellsDY_pp,
#                                         cellsDZ_pp,
#                                         ctypes.c_int(numCells),
#                                         ctypes.c_int(globalStart),
#                                         newNumCells_p
#                                         )
    print("Rank %i, After call: cellsX = " %rank, cellsX_p[:5])

#     return cellsX_pp.contents, cellsY_pp.contents, cellsZ_pp.contents, cellsDX_pp.contents, cellsDY_pp.contents, cellsDZ_pp.contents, newNumCells
#     return cellsX_p.contents, cellsY_p.contents, cellsZ_p.contents, cellsDX_p.contents, cellsDY_p.contents, cellsDZ_p.contents, newNumCells
    del cellsX
    del cellsY
    del cellsZ
    del cellsDX
    del cellsDY
    del cellsDZ
    
    return cellsX_p[:newNumCells], cellsY_p[:newNumCells], cellsZ_p[:newNumCells], cellsDX_p[:newNumCells], cellsDY_p[:newNumCells], cellsDZ_p[:newNumCells], newNumCells
#     return cellsX[:newNumCells], cellsY[:newNumCells], cellsZ[:newNumCells], cellsDX[:newNumCells], cellsDY[:newNumCells], cellsDZ[:newNumCells], newNumCells


if __name__=="__main__":
    
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    r=50
    numCells=r*(rank+1)
    
    np.random.seed(rank)
    cellsX=2*np.random.rand(numCells)-1
    cellsY=2*np.random.rand(numCells)-1
    cellsZ=2*np.random.rand(numCells)-1
    cellsDX=np.ones(numCells)
    cellsDY=np.ones(numCells)
    cellsDZ=np.ones(numCells)
    
    globalStart=0
    
    for i in range(rank):
        globalStart += r*(i+1)
    print("rank %i starts at %i and has %i cells. " %(rank,globalStart,numCells))
    
    newCellsX, newCellsY, newCellsZ, newCellsDX, newCellsDY, newCellsDZ, newNumCells = callZoltan(cellsX, cellsY, cellsZ, cellsDX, cellsDY, cellsDZ, numCells, globalStart)
#     cellsX, cellsY, cellsZ, cellsDX, cellsDY, cellsDZ, newNumCells = callZoltan(cellsX, cellsY, cellsZ, cellsDX, cellsDY, cellsDZ, numCells, globalStart)


#     cellsX=None
#     cellsY=None
#     cellsZ=None
#     cellsDX=None
#     cellsDY=None
#     cellsDZ=None
#     
#     del cellsX
#     del cellsY
#     del cellsZ
#     del cellsDX
#     del cellsDY
#     del cellsDZ
    
    xmean=np.mean(newCellsX)
    ymean=np.mean(newCellsY)
    zmean=np.mean(newCellsZ)
#     xmean=np.mean(cellsX)
#     ymean=np.mean(cellsY)
#     zmean=np.mean(cellsZ)
      
    rprint(0,"len(newCellsX) = ", len(newCellsX)) 
#     rprint(0,"len(cellsX) = ", len(cellsX)) 
    rprint(0,"newNumCells = ", newNumCells)    
    assert len(newCellsX)==newNumCells, "Length of cellsX does not equal newNumCells."  
#     assert len(cellsX)==newNumCells, "Length of cellsX does not equal newNumCells."  
    print("After load balancing, rank %i has %i cells cenetered at (x,y,z)=(%f,%f,%f)." %(rank,len(newCellsX),xmean,ymean,zmean))
#     print("After load balancing, rank %i has %i cells cenetered at (x,y,z)=(%f,%f,%f)." %(rank,len(cellsX),xmean,ymean,zmean))

    
    comm.Barrier()
    
    print("Called MPI finalize.")
    
    print("Doing some other stuff now...")
    
    

    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    color=np.random.rand(3,)
    ax.scatter(newCellsX,newCellsY,newCellsZ,'o',color=color,label="rank %i" %rank)
#     ax.scatter(cellsX,cellsY,cellsZ,'o',color=color,label="rank %i" %rank)
    ax.set_xlim([-1,1])
    ax.set_ylim([-1,1])
    ax.set_zlim([-1,1])
    plt.title("Rank %i" %rank)
#     plt.legend()
    comm.Barrier()
    plt.show()
    
    
#     input()
#     comm.Barrier()
#     MPI.Finalize()

#     input()
#     exit(0)





