"""3 dimensional tests for the Zoltan partitioner. The test follows
the same pattern as the 2D test.

To see the output from the script try the following::

    $ mpirun -np 4 python 3d_partition.py --plot

"""
import sys

import mpi4py.MPI as mpi
comm = mpi.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

from cyarray.api import UIntArray, DoubleArray
from pyzoltan.core import zoltan
from pyzoltan.core import zoltan_comm

from numpy import random
import numpy as np


colors = ['r', 'g', 'b', 'y', 'm', 'k', 'c', 'burlywood',
          'r', 'g', 'b', 'y', 'm', 'k', 'c', 'burlywood',
          'r', 'g', 'b', 'y', 'm', 'k', 'c', 'burlywood',
          'r', 'g', 'b', 'y', 'm', 'k', 'c', 'burlywood',
          'r', 'g', 'b', 'y', 'm', 'k', 'c', 'burlywood',
          'r', 'g', 'b', 'y', 'm', 'k', 'c', 'burlywood']

savedir = '/Users/nathanvaughn/Desktop/zoltan/'

def plot_points(x, y, z, slice_data, title, filename):
    if '--plot' not in sys.argv:
        return

    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    fig = plt.figure()
    s1 = fig.add_subplot(111, projection='3d')
#     s1.axes = Axes3D(fig)
    for i in range(size):
        s1.axes.plot3D(
            x[slice_data[i]], y[slice_data[i]], z[slice_data[i]],
            c=colors[i], marker='o', markersize=1, linestyle='None', alpha=0.5
        )

    s1.axes.set_xlabel( 'X' )
    s1.axes.set_ylabel( 'Y' )
    s1.axes.set_zlabel( 'Z' )

    plt.title(title)
    plt.show() 
    plt.savefig(savedir+filename)
    
def plot_points_single_proc(x, y, z, rank, title, filename):
    if '--plot' not in sys.argv:
        return

    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    fig = plt.figure()
    s1 = fig.add_subplot(111, projection='3d')
#     s1.axes = Axes3D(fig)
#     for i in range(size):
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
    plt.savefig(savedir+filename)


numPoints = 1<<6
LBMETHOD='RCB'
# LBMETHOD='HSFC'
# LBMETHOD='BLOCK'
# LBMETHOD='REFTREE'

x = random.random( numPoints )
y = random.random( numPoints )
z = random.random( numPoints )
gid = np.arange( numPoints*size, dtype=np.uint32 )[rank*numPoints:(rank+1)*numPoints]

X = np.zeros( size * numPoints )
Y = np.zeros( size * numPoints )
Z = np.zeros( size * numPoints )
GID = np.arange( numPoints*size, dtype=np.uint32 )

comm.Gather( sendbuf=x, recvbuf=X, root=0 )
comm.Gather( sendbuf=y, recvbuf=Y, root=0 )
comm.Gather( sendbuf=z, recvbuf=Z, root=0 )

## Uncomment if you want to plot initial data.  It is just random points
# if rank == 0:
#     slice_data = [slice(i*numPoints, (i+1)*numPoints) for i in range(size)]
#     plot_points(
#         X, Y, Z, slice_data, title="Initial Distribution",
#         filename="initial%i_"%size+LBMETHOD+'.pdf'
#     )

# partition the points using PyZoltan
## DoubleArrays are from cyarray.  They act more like c++ vectors.  Can be resized, can get the pointer to the array, etc.
xa = DoubleArray(numPoints); xa.set_data(x)
ya = DoubleArray(numPoints); ya.set_data(y)
za = DoubleArray(numPoints); za.set_data(z)
gida = UIntArray(numPoints); gida.set_data(gid)

# create the geometric partitioner
print("Setting up geometric partitioner.")
pz = zoltan.ZoltanGeometricPartitioner(
    dim=3, comm=comm, x=xa, y=ya, z=za, gid=gida)
print("Completed geometric partitioner setup.")
# call the load balancing function
print("Calling the load balancer.")

pz.set_lb_method(LBMETHOD)
pz.Zoltan_Set_Param('DEBUG_LEVEL', '1')
pz.Zoltan_LB_Balance()
print("Load balancer complete.")

# get the new assignments
my_global_ids = list( gid )
original_my_global_ids = np.copy(my_global_ids)

# remove points to be exported
for i in range(pz.numExport):
    my_global_ids.remove( pz.exportGlobalids[i] )
afterExport_my_global_ids = np.copy(my_global_ids)

# add points to be imported
for i in range(pz.numImport):
    my_global_ids.append( pz.importGlobalids[i] )
afterImport_my_global_ids = np.copy(my_global_ids)

exportDestinations=np.zeros(len(pz.exportProcs),dtype=np.uint32)
for i in range(len(pz.exportProcs)):
    exportDestinations[i] = pz.exportProcs[i]

## Just for understanding purposes...
for i in range(size):
    if rank==i:
        print("\nRank %i"%rank + 
              "\nOriginally owned: ", original_my_global_ids,
              "\nHeld on to:       ", afterExport_my_global_ids)
#         for j in range(len(pz.exportGlobalids)):
#             print("Exporting %i to proc %i" %(pz.exportGlobalids[j],pz.exportProcs[j]) )
            
        print("Ended up with:    ", afterImport_my_global_ids, "\n")
    comm.barrier()

new_gids = np.array( my_global_ids, dtype=np.uint32 )

# gather the new gids on root as a list
NEW_GIDS = comm.gather( new_gids, root=0 )

#save the new partition
print("Plotting final distribution.")
if rank == 0:
    plot_points(
        X, Y, Z, NEW_GIDS,
        title='Final Distribution', filename='final%i_'%size+LBMETHOD+'.pdf'
    )

comm.barrier()
print("Plotted final distribution.")


## Communicate the changes

# create the ZComm object
nsend=len(pz.exportProcs)
tag1 = np.int32(1)
tag2 = np.int32(2)
tag3 = np.int32(3)
zcomm1 = zoltan_comm.ZComm(comm, tag=tag1, nsend=nsend, proclist=pz.exportProcs.get_npy_array())
# zcomm2 = zoltan_comm.ZComm(comm, tag=tag2, nsend=nsend, proclist=pz.exportProcs.get_npy_array())
# zcomm3 = zoltan_comm.ZComm(comm, tag=tag3, nsend=nsend, proclist=pz.exportProcs.get_npy_array())

# the data to send and receive
send_x=np.zeros(nsend)
send_y=np.zeros(nsend)
send_z=np.zeros(nsend)
for i in range(nsend):
    send_x[i] = x[ pz.exportGlobalids[i] - rank*numPoints ]
    send_y[i] = y[ pz.exportGlobalids[i] - rank*numPoints ]
    send_z[i] = z[ pz.exportGlobalids[i] - rank*numPoints ]
    
recv_x = np.ones( zcomm1.nreturn )
recv_y = np.ones( zcomm1.nreturn )
recv_z = np.ones( zcomm1.nreturn )

# use zoltan to exchange doubles
# print("Proc %d, Sending %s to %s"%(rank, send_x, pz.exportProcs.get_npy_array()))
zcomm1.Comm_Do(send_x, recv_x)
zcomm1.Comm_Do(send_y, recv_y)
zcomm1.Comm_Do(send_z, recv_z)
# print("Proc %d, Received %s"%(rank, recv_x))
# print("Proc %d, Received %s"%(rank, recv_y))
# print("Proc %d, Received %s"%(rank, recv_z))

original_x = np.zeros(len(afterExport_my_global_ids))
original_y = np.zeros(len(afterExport_my_global_ids))
original_z = np.zeros(len(afterExport_my_global_ids))
for i in range(len(afterExport_my_global_ids)):
    original_x[i] = x[ afterExport_my_global_ids[i] - rank*numPoints ]
    original_y[i] = y[ afterExport_my_global_ids[i] - rank*numPoints ]
    original_z[i] = z[ afterExport_my_global_ids[i] - rank*numPoints ]
balanced_x = np.append( original_x, np.copy(recv_x))
balanced_y = np.append( original_y, np.copy(recv_y))
balanced_z = np.append( original_z, np.copy(recv_z))


assert len(balanced_x)==len(balanced_y)


comm.barrier()
print("Processor %i now owns %i particles." %(rank,len(balanced_x)))
plot_points_single_proc(
    balanced_x, balanced_y, balanced_z, rank,
    title='Final Distribution on Rank %i'%rank, filename='final%i_'%size+LBMETHOD+'_rank_%i.pdf'%rank
)
