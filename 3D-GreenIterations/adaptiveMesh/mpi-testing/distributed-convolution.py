# import mpi4py
# mpi4py.rc.initialize = False
# mpi4py.rc.finalize = False
import sys
sys.path.insert(1, '../ctypesTests/')

import numpy as np
import time
from numpy import float32, float64, int32
import array
import pickle
import treecodeWrappers_distributed
from mpi4py import MPI



def global_dot(u,v,comm):
    local_dot = np.dot(u,v)
    global_dot = comm.allreduce(local_dot)
    return global_dot
    
    


def mpiRun(numPoints):
        

    
    comm = MPI.COMM_WORLD
    
    if comm.size==6:
        nx=2
        ny=3
        nz=1
    elif comm.size==4:
        nx=2
        ny=2
        nz=1
    elif comm.size==2:
        nx=2
        ny=1
        nz=1
    elif comm.size==1:
        nx=1
        ny=1
        nz=1
    else: 
        print('Not prepared for comm.size = ', comm.size)
        return
        
    
    assert nx*ny*nz==comm.size
    
    
    numProcs = comm.size
    rank = comm.Get_rank()
    if (rank==0): print("Creating a %d x %d x %d processor grid..." % (nx, ny, nz) )

    
    
    local_x_points = -(-numPoints // nx)
    local_y_points = -(-numPoints // ny)
    local_z_points = -(-numPoints // nz)
    if rank==0:
        local_x_points += (numPoints - nx*local_x_points) 
        local_y_points += (numPoints - ny*local_y_points) 
        local_z_points += (numPoints - nz*local_z_points) 
#     print('Rank %i has (%i,%i,%i) points in x, y, and z dimension.' %(rank,local_x_points,local_y_points,local_z_points) )
    
    
    ccomm = comm.Create_cart(dims = (nx, ny, nz),periods =(False,False,False),reorder=False)
    coord3d = ccomm.Get_coords(rank)
    
    xlow = 1+coord3d[0]/nx
    xhigh = 1+(coord3d[0]+1)/nx
    xvec = np.linspace(xlow,xhigh,local_x_points,endpoint=False)
    
#     print(xvec)
    
    ylow = 1+coord3d[1]/ny
    yhigh = 1+(coord3d[1]+1)/ny
    yvec = np.linspace(ylow,yhigh,local_y_points,endpoint=False)
    
    zlow = 1+coord3d[2]/nz
    zhigh = 1+(coord3d[2]+1)/nz
    zvec = np.linspace(zlow,zhigh,local_z_points,endpoint=False)
    
    X,Y,Z = np.meshgrid(xvec,yvec,zvec)
    X = X.flatten()
    Y = Y.flatten()
    Z = Z.flatten()
#     RHO = np.random.rand(len(X))
    RHO = np.exp( - np.sqrt(X*X + Y*Y + Z*Z))
    W = np.ones(len(X))
    
#     print(np.shape(X))
#     print(RHO)
    
    
#     print("rank %i: x,y,z,rho,w = %f,%f,%f,%f,%f" %(rank,X[0],Y[0],Z[0],RHO[0],W[0]))
 
    
    treecodeOrder=8
    theta=0.8
    maxParNode=20
    batchSize=20
    GPUpresent=False
    potentialType=2
    gaussianAlpha=0.5
    
    nPoints = len(X)
    print ("In 3D topology, Processor ",rank, " has coordinates ",coord3d, " and contains ", nPoints, " points.")

    
#     print("Before calling treecode wrapper, is MPI initialized in python? True for yes, False for no:", MPI.Is_initialized())
#     print(MPI.Is_initialized())
#     output=np.zeros(len(X))

    ## Compute reference values (compute with theta=0.0)
    referenceOutput = treecodeWrappers_distributed.callTreedriver(nPoints, nPoints, 
                   np.copy(X), np.copy(Y), np.copy(Z), np.copy(RHO), 
                   np.copy(X), np.copy(Y), np.copy(Z), np.copy(RHO), np.copy(W),
                   potentialType, gaussianAlpha, treecodeOrder, 0.0, maxParNode, batchSize, GPUpresent)

    startTime = MPI.Wtime()
    output = treecodeWrappers_distributed.callTreedriver(nPoints, nPoints, 
                   np.copy(X), np.copy(Y), np.copy(Z), np.copy(RHO), 
                   np.copy(X), np.copy(Y), np.copy(Z), np.copy(RHO), np.copy(W),
                   potentialType, gaussianAlpha, treecodeOrder, theta, maxParNode, batchSize, GPUpresent)
    endTime = MPI.Wtime()
    if rank==0:
        print("Convolution time for %i points using %i ranks: %f seconds" %(numPoints**3,numProcs,endTime-startTime))
#     print("Shape of output: ", np.shape(output))     
#     print(output)
    if np.isnan(output).any(): 
        print("Output array contains NaN on rank %i." %rank)
    
    
    L2err = np.sqrt( global_dot(output-referenceOutput,output-referenceOutput,comm) )
    L2err /= np.sqrt( global_dot(referenceOutput,referenceOutput,comm) )
    if rank==0: print("Relative L2 error: ", L2err ) 
    
    outputSum = global_dot(output,output,comm)
    if rank==0: print("Sum of output squared: ", outputSum)
    
#     return
  

if __name__=="__main__":    
    
    mpiRun(12)
    