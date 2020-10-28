import sys
import mpi4py.MPI as MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

from cyarray.api import UIntArray, DoubleArray
from pyzoltan.core import zoltan
from pyzoltan.core import zoltan_comm

from loadBalancer import loadBalance


def buildAndDistributeMesh(numProcs, scheme, refinementParameter):
    '''
    This function sets up the parallel mesh building and redistributes the points.  
    It uses some number of processors to build initial mesh, then distributes to all of the processors.
    :param numProcs:
    :param scheme:
    :param refinementParameter:
    '''
    
    
    #     Each processor involved in the build needs to create a root cell.
    '''
    How do I want to do this?  
    1) Could divide domain into rectangular chunks, then have each processor own a chunk and make a root cell.
    2) Could do some minimum level of refinement, then distribute those cells over all available processors.
    3) Could let 1 proc start, it generates 8 children.  Distribute to other procs.  Let them continue to distribute until all procs are working on some cells.  
    
    '''
    return x, y, z, w


if __name__=="__main__":
    
    np = 6
    nx = 2
    ny = 3
    nz = 1
    
    xL = 20
    yL = 20
    zL = 20
    
    buildAndDistributeMesh(np,'LW5', 500)