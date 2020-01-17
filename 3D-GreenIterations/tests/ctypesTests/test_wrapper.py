import numpy as np
import ctypes
import sys
from time import time

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

sys.path.insert(1, '/Users/nathanvaughn/Documents/GitHub/TAGI/3D-GreenIterations/src/utilities')
sys.path.insert(1, '/home/njvaughn/TAGI/3D-GreenIterations/src/utilities')


from mpiUtilities import global_dot, scatterArrays, rprint


try:
    import treecodeWrappers_distributed as treecodeWrappers
except ImportError:
    rprint(rank,'Unable to import treecodeWrapper due to ImportError')
except OSError:
    print('Unable to import treecodeWrapper due to OSError')
    import treecodeWrappers_distributed as treecodeWrappers
    
    
N=100000
verbosity=0
GPU = int(sys.argv[1]);
if GPU==1:
    GPUpresent=True
else:
    GPUpresent=False
    

X=np.random.rand(N)
Y=np.random.rand(N)
Z=np.random.rand(N)
Q=np.random.rand(N)
W=np.random.rand(N)

print(Z[0])
print("Global dots: ", global_dot(X,X,comm))


kernelName = "coulomb"
approximationName = "lagrange"
singularityHandling = "subtraction"
treecodeOrder=2
theta=0.0
maxParNode=4000
batchSize=4000
gaussianAlpha=1.0
print("Calling treecode wrapper for first time:")
start=time()
V1 = treecodeWrappers.callTreedriver(  N, N, 
                                       np.copy(X), np.copy(Y), np.copy(Z), np.copy(Q), 
                                       np.copy(X), np.copy(Y), np.copy(Z), np.copy(Q), np.copy(W),
                                       kernelName, gaussianAlpha, singularityHandling, approximationName,
                                       treecodeOrder, theta, maxParNode, batchSize, GPUpresent, verbosity)
end=time()
print("Treecode took %f seconds" %(end-start))

# rprint(rank,"Completed first call to wrapper.")
# rprint(rank,"Does X still exist? Size?", np.size(X))
# 
# M = X+Y+Z+Q+W
# 
# 
# V2 = treecodeWrappers.callTreedriver(  N, N, 
#                                        X,Y,Z,Q, 
#                                        X,Y,Z,Q,W,
#                                        kernelName, gaussianAlpha, singularityHandling, approximationName,
#                                        treecodeOrder, theta, maxParNode, batchSize, GPUpresent)
#  
# rprint(rank,"Completed second call to wrapper.")
# rprint(rank,"Max error: ", np.max( np.abs(V2-V1)))
# rprint(rank,"Does X still exist? Size?", np.size(X))





