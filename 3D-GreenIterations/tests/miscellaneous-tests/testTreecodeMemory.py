'''
'''
import os
import sys
import time
import inspect
import resource
import unittest
import numpy as np
import pylibxc
import itertools
import csv
from scipy.optimize import anderson
from scipy.optimize import root as scipyRoot
from scipy.special import sph_harm
from pyevtk.hl import unstructuredGridToVTK
from pyevtk.vtk import VtkTriangle, VtkQuad, VtkPolygon, VtkVoxel, VtkHexahedron
import mpi4py.MPI as MPI




sys.path.insert(1, '/Users/nathanvaughn/Documents/GitHub/TAGI/3D-GreenIterations/src/utilities')
sys.path.insert(1, '/Users/nathanvaughn/Documents/GitHub/TAGI/3D-GreenIterations/src/dataStructures')
sys.path.insert(1, '/home/njvaughn/TAGI/3D-GreenIterations/src/utilities')
from loadBalancer import loadBalance
from mpiUtilities import global_dot, scatterArrays, rprint
from mpiMeshBuilding import  buildMeshFromMinimumDepthCells
try:
    import treecodeWrappers_distributed as treecodeWrappers
except ImportError:
    rprint(rank,'Unable to import treecodeWrapper due to ImportError')
except OSError:
    print('Unable to import treecodeWrapper due to OSError')
    import treecodeWrappers_distributed as treecodeWrappers


if __name__=="__main__":
    maxParNode=500
    batchSize=500
    GPUpresent=False
    theta=0.8
    treecodeOrder=7
    gaussianAlpha=1.0
    
    
    N=10000
    n=10
    RHO = np.random.rand(N)
    X = np.random.rand(N)
    Y = np.random.rand(N)
    Z = np.random.rand(N)
    W = np.ones(N)
    
    initialMemory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    previousMemory=initialMemory
    print('INITIAL MEMORY USAGE: ', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss )
    
    for i in range(10):
        kernelName = "coulomb"
        approximationName = "lagrange"
        singularityHandling = "subtraction"
        verbosity=0
        V_hartreeNew = treecodeWrappers.callTreedriver(N, N, 
                                                       np.copy(X), np.copy(Y), np.copy(Z), np.copy(RHO), 
                                                       np.copy(X), np.copy(Y), np.copy(Z), np.copy(RHO), np.copy(W),
                                                       kernelName, gaussianAlpha, singularityHandling, approximationName,
                                                       treecodeOrder, theta, maxParNode, batchSize, GPUpresent, verbosity)
        
#         V_hartreeNew = treecodeWrappers.callTreedriver(N, N, 
#                                                        X, Y, Z, RHO, 
#                                                        X, Y, Z, RHO, W,
#                                                        kernelName, gaussianAlpha, singularityHandling, approximationName,
#                                                        treecodeOrder, theta, maxParNode, batchSize, GPUpresent, verbosity)
    
        newMemory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        print("Memory growth this iteration, %i to %i: " %(previousMemory,newMemory))
        previousMemory=newMemory
        
    finalMemory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print("Memory growth over %i iterations, %i to %i :" %(n,initialMemory,finalMemory))
        



