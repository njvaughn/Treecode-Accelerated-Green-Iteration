import numpy as np
import os
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size() 


srcdir="/home/njvaughn/TAGI/3D-GreenIterations/src/"
sys.path.append(srcdir+'dataStructures')
sys.path.append(srcdir+'Green-Iteration-Routines')
sys.path.append(srcdir+'utilities')
sys.path.append(srcdir+'../ctypesTests/src')

sys.path.append(srcdir+'../ctypesTests')
sys.path.append(srcdir+'../ctypesTests/lib')

sys.path.append('../src/dataStructures')
sys.path.append('../src/utilities') 

import BaryTreeInterface as BT
from mpiUtilities import global_dot, rprint 



if __name__=="__main__":
    
    start=time.time()
    fileBase="/home/njvaughn/PSPmesh/"
#     nPoints="617000"
    nPoints="1198000"
#     nPoints="2070144"
    X = np.load(fileBase+"X_"+nPoints+".npy")
    Y = np.load(fileBase+"Y_"+nPoints+".npy")
    Z = np.load(fileBase+"Z_"+nPoints+".npy")
    W = np.load(fileBase+"W_"+nPoints+".npy")
#     RHO = np.loadtxt(fileBase+"RHO_"+nPoints+".txt")
    r = np.sqrt(X*X + Y*Y + Z*Z)
    RHO = np.exp(-r)
    
#     print(RHO[:5])
#     print(W[:5])
    end=time.time()
    print("Time to load in data: %f seconds" %(end-start))
    
    
    
    treecodeOrder=7
    theta=0.0
    maxPerTargetLeaf=2000
    maxPerSourceLeaf=2000 
    GPUpresent=True
    treecode_verbosity=1
    
    approximation = BT.Approximation.LAGRANGE
#     singularity   = BT.Singularity.SUBTRACTION
    singularity   = BT.Singularity.SKIPPING
#     computeType   = BT.ComputeType.PARTICLE_CLUSTER
    computeType   = BT.ComputeType.CLUSTER_CLUSTER
#     computeType   = BT.ComputeType.PARTICLE_CLUSTER
    
    kernel = BT.Kernel.COULOMB
    numberOfKernelParameters = 0
    kernelParameters = np.array([0.5])
    
    
    ## Either compute and save, or load V_direct
#     #compute and save
 
    try:
        V_direct = np.load(fileBase+"V_direct_"+nPoints+".npy")
    except FileNotFoundError:
        print("V_direct not found.  Computing it now.")
     
        start=time.time()
        V_direct = BT.callTreedriver(  len(X), len(X), 
                                       X, Y, Z, RHO, 
                                       np.copy(X), np.copy(Y), np.copy(Z), np.copy(RHO), np.copy(W),
                                       kernel, numberOfKernelParameters, kernelParameters, 
                                       singularity, approximation, computeType,
                                       treecodeOrder, theta, maxPerSourceLeaf, maxPerTargetLeaf, 
                                       GPUpresent, treecode_verbosity)
 
        end=time.time()
        print("Direct sum took %f seconds." %(end-start))
        np.save(fileBase+"V_direct_"+nPoints, V_direct)
     
 
     
#     theta=0.8
#     treecode_verbosity=0
#     while True:
#         print("\n\nEnter a Lagrange treecode interpolation degree:")
#         treecodeOrder=int( input() )
#         print("Enter a sizeCheck:")
#         sizeCheck=float( input() )
#         if treecodeOrder==0:
#             break
#         start=time.time()
#         V_tree = BT.callTreedriver(  len(X), len(X), 
#                                        X, Y, Z, RHO, 
#                                        np.copy(X), np.copy(Y), np.copy(Z), np.copy(RHO), np.copy(W),
#                                        kernel, numberOfKernelParameters, kernelParameters, 
#                                        singularity, approximation, computeType,
#                                        treecodeOrder, theta, maxPerSourceLeaf, maxPerTargetLeaf, 
#                                        GPUpresent, treecode_verbosity,sizeCheck=sizeCheck)
#         end=time.time()
#         L2error = np.sqrt(np.sum( (V_direct-V_tree)**2*W ))
#         print("Convolution time for order %i took %f seconds with error %1.3e" %(treecodeOrder,end-start,L2error))
#        
#        
#     while True:
#         approximation = BT.Approximation.HERMITE
#         print("\n\nEnter a Hermite treecode interpolation degree:")
#         treecodeOrder=int( input() )
#         print("Enter a sizeCheck:")
#         sizeCheck=float( input() )
#         if treecodeOrder==0:
#             break
#         start=time.time()
#         V_tree = BT.callTreedriver(  len(X), len(X), 
#                                        X, Y, Z, RHO, 
#                                        np.copy(X), np.copy(Y), np.copy(Z), np.copy(RHO), np.copy(W),
#                                        kernel, numberOfKernelParameters, kernelParameters, 
#                                        singularity, approximation, computeType,
#                                        treecodeOrder, theta, maxPerSourceLeaf, maxPerTargetLeaf, 
#                                        GPUpresent, treecode_verbosity,sizeCheck=sizeCheck)
#         end=time.time()
#         L2error = np.sqrt(np.sum( (V_direct-V_tree)**2*W ))
#         print("Convolution time for order %i took %f seconds with error %1.3e" %(treecodeOrder,end-start,L2error))
#     
#     
#     
#     # Test sizing
#     approximation = BT.Approximation.LAGRANGE
# #     approximation = BT.Approximation.HERMITE
#     treecodeOrder=7
#     theta=0.8
#     treecode_verbosity=0
#     for maxPerTargetLeaf in [1000, 2000, 4000, 8000]:
#         for maxPerSourceLeaf in [1000, 2000, 4000, 8000]:
#             
#             
#             print()
#             print("maxPerTargetLeaf = ", maxPerTargetLeaf)
#             print("maxPerSourceLeaf = ", maxPerSourceLeaf)
#             start=time.time()
#             V_tree = BT.callTreedriver(len(X), len(X), 
#                                        np.copy(X), np.copy(Y), np.copy(Z), np.copy(RHO), 
#                                        X, Y, Z, RHO, W,
#                                        kernel, numberOfKernelParameters, kernelParameters, 
#                                        singularity, approximation, computeType,
#                                        treecodeOrder, theta, maxPerSourceLeaf, maxPerTargetLeaf, 
#                                        GPUpresent, treecode_verbosity)
#             end=time.time()
#             L2error = np.sqrt(np.sum( (V_direct-V_tree)**2*W ))
#               
#             print("time       = %1.3f" %(end-start))
#             print("error      = %1.3e" %L2error)
             
        
    
#     # Test params
    computeType   = BT.ComputeType.PARTICLE_CLUSTER
    batchSize=2000
    maxParNode=2000
    treecode_verbosity=0
    approximation = BT.Approximation.LAGRANGE
      
    timesH=[]
    errorsH=[]
    for theta in [0.6, 0.7, 0.8, 0.9]:
        for treecodeOrder in [3, 4, 5, 6]:
              
            print()
            print("theta         = ", theta)
            print("treecodeOrder = ", treecodeOrder)
              
            start=time.time()
            V_tree = BT.callTreedriver(  len(X), len(X), 
                                       X, Y, Z, RHO, 
                                       np.copy(X), np.copy(Y), np.copy(Z), np.copy(RHO), np.copy(W),
                                       kernel, numberOfKernelParameters, kernelParameters, 
                                       singularity, approximation, computeType,
                                       treecodeOrder, theta, maxPerSourceLeaf, maxPerTargetLeaf, 
                                       GPUpresent, treecode_verbosity)
            end=time.time()
            timeH=(end-start)
            L2error = np.sqrt(np.sum( (V_direct-V_tree)**2*W ))
            print("time          = %1.3f" %timeH)
            print("error         = %1.3e" %L2error)
              
            timesH.append(timeH)
            errorsH.append(L2error)
              
#     approximation = BT.Approximation.LAGRANGE
    computeType   = BT.ComputeType.CLUSTER_CLUSTER
    timesL=[]
    errorsL=[]
    for theta in [0.6, 0.7, 0.8, 0.9]:
        for treecodeOrder in [3, 5, 7, 9]:
              
            print()
            print("theta         = ", theta)
            print("treecodeOrder = ", treecodeOrder)
              
            start=time.time()
            V_tree = BT.callTreedriver(  len(X), len(X), 
                                       X, Y, Z, RHO, 
                                       np.copy(X), np.copy(Y), np.copy(Z), np.copy(RHO), np.copy(W),
                                       kernel, numberOfKernelParameters, kernelParameters, 
                                       singularity, approximation, computeType,
                                       treecodeOrder, theta, maxPerSourceLeaf, maxPerTargetLeaf, 
                                       GPUpresent, treecode_verbosity)
            end=time.time()
            timeL=(end-start)
            L2error = np.sqrt(np.sum( (V_direct-V_tree)**2*W ))
            print("time          = %1.3f" %timeL)
            print("error         = %1.3e" %L2error)
              
            timesL.append(timeL)
            errorsL.append(L2error)
      
      
    plt.loglog(errorsL,timesL,'go',label="Lagrange")
    plt.loglog(errorsH,timesH,'bo',label="Hermite")
    plt.xlabel("Error")
    plt.ylabel("Time")
    plt.legend()
    plt.grid()
    plt.title("Yukawa Hermite vs. Lagrange, Mesh N="+nPoints )
    plt.savefig(fileBase+"curves_yukawa_"+nPoints)
#     plt.show()
#              
# #         
#     
#     
#     
#     
#     
#     
#     
#     
#     
#     
