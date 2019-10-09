import sys
import mpi4py.MPI as MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

from cyarray.api import UIntArray, DoubleArray
from pyzoltan.core import zoltan
from pyzoltan.core import zoltan_comm

import numpy as np


colors = ['r', 'g', 'b', 'y', 'm', 'k', 'c', 'burlywood',
          'r', 'g', 'b', 'y', 'm', 'k', 'c', 'burlywood',
          'r', 'g', 'b', 'y', 'm', 'k', 'c', 'burlywood',
          'r', 'g', 'b', 'y', 'm', 'k', 'c', 'burlywood',
          'r', 'g', 'b', 'y', 'm', 'k', 'c', 'burlywood',
          'r', 'g', 'b', 'y', 'm', 'k', 'c', 'burlywood']


def plot_points_single_proc(x, y, z, rank, title):
    if '--plot' not in sys.argv:
        return

    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    fig = plt.figure()
    s1 = fig.add_subplot(111, projection='3d')
    s1.axes.plot3D(
        x, y, z,
        c=colors[rank], marker='o', markersize=1, linestyle='None', alpha=0.5
    )

    s1.axes.set_xlabel( 'X' )
    s1.axes.set_ylabel( 'Y' )
    s1.axes.set_zlabel( 'Z' )
    
    s1.axes.set_xlim([0,1])
    s1.axes.set_ylim([0,1])
    s1.axes.set_zlim([0,1])

    plt.title(title)
    plt.show() 
#     plt.savefig(savedir+filename)


def loadBalance(x,y,z,data,verbosity=0):
    '''
    Each processor calls loadBalance.  Using zoltan, the particles are balanced and redistributed as necessary.  Returns the balanced arrays
    '''
    
    initialNumPoints = len(x)
    pointsOnEachProc = comm.allgather(initialNumPoints)
    
    # Compute local offset
    localOffset = np.int( np.sum( pointsOnEachProc[:rank]) )   
    
    
    if verbosity>0:
        if rank==0: print("Points on each proc: ", pointsOnEachProc)
        print("Rank %i offset: %i" %(rank,localOffset))
        print("Rank %i initialNumPoints: %i" %(rank,initialNumPoints))
        
    
    # Begin load balancing
    globalNumPoints = np.sum(pointsOnEachProc)
    if ( (verbosity>0) and (rank==0) ): print("global number of points: ", globalNumPoints)
    gid = np.arange( globalNumPoints, dtype=np.uint32 )[localOffset:localOffset+initialNumPoints]
    
    if verbosity>0: print("rank %i, gid: " %(rank), gid)

    ## DoubleArrays are from cyarray.  They act more like c++ vectors.  Can be resized, can get the pointer to the array, etc.
    xa = DoubleArray(initialNumPoints); xa.set_data(x)
    ya = DoubleArray(initialNumPoints); ya.set_data(y)
    za = DoubleArray(initialNumPoints); za.set_data(z)
    gida = UIntArray(initialNumPoints); gida.set_data(gid)
    # create the geometric partitioner
    LBMETHOD='HSFC'
    if ( (verbosity>0) and (rank==0) ): print("Setting up geometric partitioner.")
    pz = zoltan.ZoltanGeometricPartitioner(
        dim=3, comm=comm, x=xa, y=ya, z=za, gid=gida)
    if ( (verbosity>0) and (rank==0) ): print("Completed geometric partitioner setup.")
    # call the load balancing function
    if ( (verbosity>0) and (rank==0) ): print("Calling the load balancer.")
    
    pz.set_lb_method(LBMETHOD)
    pz.Zoltan_Set_Param('DEBUG_LEVEL', '0')
    pz.Zoltan_LB_Balance()
    if ( (verbosity>0) and (rank==0) ): print("Load balancer complete.")
    
    # get the new assignments
    my_global_ids = list( gid )
    original_my_global_ids = np.copy(my_global_ids)
    
    # remove points to be exported
    for i in range(pz.numExport):
        my_global_ids.remove( pz.exportGlobalids[i] )
    afterExport_my_global_ids = np.copy(my_global_ids)
    
    comm.barrier()    
    ## Communicate the changes
    
    # create the ZComm object
    nsend=len(pz.exportProcs)
    tag = np.int32(1)
    zcomm = zoltan_comm.ZComm(comm, tag=tag, nsend=nsend, proclist=pz.exportProcs.get_npy_array())
    
    # the data to send and receive
    send_x=np.zeros(nsend)
    send_y=np.zeros(nsend)
    send_z=np.zeros(nsend)
    send_data=np.zeros(nsend)
    for i in range(nsend):
        send_x[i] = x[ pz.exportGlobalids[i] - localOffset ]
        send_y[i] = y[ pz.exportGlobalids[i] - localOffset ]
        send_z[i] = z[ pz.exportGlobalids[i] - localOffset ]
        send_data[i] = data[ pz.exportGlobalids[i] - localOffset ]
        
    recv_x = np.ones( zcomm.nreturn )
    recv_y = np.ones( zcomm.nreturn )
    recv_z = np.ones( zcomm.nreturn )
    recv_data = np.ones( zcomm.nreturn )
    
    # use zoltan to exchange doubles
    zcomm.Comm_Do(send_x, recv_x)
    zcomm.Comm_Do(send_y, recv_y)
    zcomm.Comm_Do(send_z, recv_z)
    zcomm.Comm_Do(send_data, recv_data)

    
    original_x = np.zeros(len(afterExport_my_global_ids))
    original_y = np.zeros(len(afterExport_my_global_ids))
    original_z = np.zeros(len(afterExport_my_global_ids))
    original_data = np.zeros(len(afterExport_my_global_ids))
    for i in range(len(afterExport_my_global_ids)):
        original_x[i] = x[ afterExport_my_global_ids[i] - localOffset ]
        original_y[i] = y[ afterExport_my_global_ids[i] - localOffset ]
        original_z[i] = z[ afterExport_my_global_ids[i] - localOffset ]
        original_data[i] = data[ afterExport_my_global_ids[i] - localOffset ]
    balanced_x = np.append( original_x, np.copy(recv_x))
    balanced_y = np.append( original_y, np.copy(recv_y))
    balanced_z = np.append( original_z, np.copy(recv_z))
    balanced_data = np.append( original_data, np.copy(recv_data))
    
    print("Rank %i started with %i points.  After load balancing it has %i points." %(rank,initialNumPoints,len(balanced_x)))
#     return
    return balanced_x,balanced_y,balanced_z, balanced_data


if __name__=="__main__":
    n = 500*(rank+1)**2
#     print(n)
    x = np.random.random( n )
    y = np.random.random( n )
    z = np.random.random( n )
    data = np.random.random( n )
    
    
    initSumX = comm.allreduce(np.sum(x))
    initSumY = comm.allreduce(np.sum(y))
    initSumZ = comm.allreduce(np.sum(z))
    initSumData = comm.allreduce(np.sum(data))
    
    plot_points_single_proc(x,y,z,rank,'Initial points for rank %i'%rank)
    x,y,z,data = loadBalance(x,y,z,data,verbosity=0)
    plot_points_single_proc(x,y,z,rank,'Final points for rank %i'%rank)
    
    finalSumX = comm.allreduce(np.sum(x))
    finalSumY = comm.allreduce(np.sum(y))
    finalSumZ = comm.allreduce(np.sum(z))
    finalSumData = comm.allreduce(np.sum(data))
    
    assert abs( (initSumX-finalSumX)/initSumX )<1e-12, "Sum over x positions not preserved."
    assert abs( (initSumY-finalSumY)/initSumY )<1e-12, "Sum over y positions not preserved."
    assert abs( (initSumZ-finalSumZ)/initSumZ )<1e-12, "Sum over z positions not preserved."
    assert abs( (initSumData-finalSumData)/initSumData )<1e-12, "Sum over data values not preserved."
    
    

    