# Imports
import mpi4py.MPI as mpi
from cyarray.api import UIntArray, DoubleArray
from pyzoltan.core import zoltan

from numpy import random
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

comm = mpi.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

print("Rank %i of size %i" %(rank,size))

colors = ['r', 'g', 'b', 'y', 'm', 'k', 'c', 'burlywood']


numPoints = 1<<12
print(numPoints)

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


print("Gathers complete.")


xa = DoubleArray(numPoints); xa.set_data(x)
ya = DoubleArray(numPoints); ya.set_data(y)
za = DoubleArray(numPoints); za.set_data(z)
gida = UIntArray(numPoints); gida.set_data(gid)

# print(xa)


print("Beginning geometic partitioner")
pz = zoltan.ZoltanGeometricPartitioner(
    dim=3, comm=comm, x=xa, y=ya, z=za, gid=gida)
print("Completed geometic partitioner")
pz.set_lb_method('RCB')
pz.Zoltan_Set_Param('DEBUG_LEVEL', '1')

pz.Zoltan_LB_Balance()