'''
Created on Jun 25, 2018

@author: nathanvaughn
'''
import matplotlib
matplotlib.use('Qt4Agg')
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import csv
import sys
import os
import mpi4py.MPI as MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


# srcdir="/Users/nathanvaughn/Documents/GitHub/TAGI/3D-GreenIterations/src/"
srcdir="/home/njvaughn/TAGI/3D-GreenIterations/src/"
sys.path.append(srcdir+'dataStructures')
sys.path.append(srcdir+'Green-Iteration-Routines')
sys.path.append(srcdir+'utilities')
sys.path.append(srcdir+'../ctypesTests/src')

sys.path.append(srcdir+'../ctypesTests')
sys.path.append(srcdir+'../ctypesTests/lib')

sys.path.append('../src/dataStructures')
sys.path.append('../src/utilities')
from mpiUtilities import global_dot, rprint
from initializationRoutines import initializeDensityFromAtomicDataExternally
import BaryTreeInterface as BT

import itertools
import time
import numpy as np
# import dask.array as da
import bisect
from pyevtk.hl import pointsToVTK
try:
    from pyevtk.hl import pointsToVTK
except ImportError:
    sys.path.append('/home/njvaughn')
    try:
        from pyevtk.hl import pointsToVTK
    except ImportError:
        print("Wasn't able to import pyevtk.hl.pointsToVTK, even after appending '/home/njvaughn' to path.")
    pass

from pyevtk.hl import unstructuredGridToVTK
from pyevtk.vtk import VtkTriangle, VtkQuad, VtkPolygon, VtkVoxel, VtkHexahedron


sys.path.insert(1, '/Users/nathanvaughn/Documents/GitHub/TAGI/3D-GreenIterations/src/utilities')
sys.path.insert(1, '/Users/nathanvaughn/Documents/GitHub/TAGI/3D-GreenIterations/src/dataStructures')
sys.path.insert(1, '/home/njvaughn/TAGI/3D-GreenIterations/src/utilities')
from loadBalancer import loadBalance
from mpiUtilities import global_dot, scatterArrays, rprint

from mpiMeshBuilding import buildMeshFromMinimumDepthCells
ThreeByThreeByThree = [element for element in itertools.product(range(3),range(3),range(3))]

from TreeStruct_CC import Tree

def find(a, x):
    'Locate the leftmost value exactly equal to x'
    i = bisect.bisect_left(a, [x,])
    if i != len(a) and a[i][0] == x:
        return i
    raise ValueError




def exportMeshForParaview(domainSize,maxSideLength,coreRepresentation, 
                          inputFile,outputFile,srcdir,order,gaugeShift,
                          divideCriterion,divideParameter1,divideParameter2,divideParameter3,divideParameter4):    
    
    
#     [coordinateFile, DummyOutputFile] = np.genfromtxt(inputFile,dtype="|U100")[:2]
    [coordinateFile, referenceEigenvaluesFile, DummyOutputFile] = np.genfromtxt(inputFile,dtype="|U100")[:3]
    [Eband, Ekinetic, Eexchange, Ecorrelation, Eelectrostatic, Etotal] = np.genfromtxt(inputFile)[3:]
#     nElectrons = int(nElectrons)
#     nOrbitals = int(nOrbitals)

#     print('Reading atomic coordinates from: ', coordinateFile)
#     atomData = np.genfromtxt(coordinateFile,delimiter=',',dtype=float)
#     if np.shape(atomData)==(5,):
#         nElectrons = atomData[3]
#     else:
#         nElectrons = 0
#         for i in range(len(atomData)):
#             nElectrons += atomData[i,3]
    
    print([coordinateFile, Etotal, Eexchange, Ecorrelation, Eband])
#     tree = Tree(-domain,domain,order,-domain,domain,order,-domain,domain,order,nElectrons,nOrbitals,additionalDepthAtAtoms=additionalDepthAtAtoms,minDepth=minDepth,gaugeShift=gaugeShift,
#                 coordinateFile=coordinateFile,smoothingEps=smoothingEpsilon,inputFile=inputFile)#, iterationOutFile=outputFile)
# 
#     
#     print('max depth ', maxDepth)
# #     tree.minimalBuildTree( maxLevels=maxDepth, initializationType='atomic',divideCriterion=divideCriterion, 
#     tree.buildTree( maxLevels=maxDepth, initializationType='atomic',divideCriterion=divideCriterion, 
#                     divideParameter1=divideParameter1, divideParameter2=divideParameter2, divideParameter3=divideParameter3, divideParameter4=divideParameter4, 
#                     savedMesh=savedMesh, printTreeProperties=True,onlyFillOne=False)
#     
#     X,Y,Z,W,RHO, XV, YV, ZV, vertexIdx, centerIdx, ghostCells = tree.extractXYZ_connected()
    
    # Set domain to be an integrer number of far-field cells
    
    
    
#     domainSize+=1/2*(maxSideLength-(2*domainSize)%maxSideLength)
#     rprint(rank,"Max side length: ",maxSideLength)
#     rprint(rank,"Domain length after adjustment: ", domainSize)
#     rprint(rank," Far field nx, ny, nz = ", 2*domainSize/maxSideLength)
    X,Y,Z,W,Xf,Yf,Zf,Wf,pointsPerCell_coarse, pointsPerCell_fine, atoms,PSPs,nPoints,nOrbitals,nElectrons,referenceEigenvalues = buildMeshFromMinimumDepthCells(domainSize,domainSize,domainSize,maxSideLength,coreRepresentation,
                                                                                                     inputFile,outputFile,srcdir,order,order,gaugeShift,
                                                                                                     MESHTYPE,MESHPARAM1,MESHPARAM2,MESHPARAM3,MESHPARAM4,saveTree=False)
   
    RHO = initializeDensityFromAtomicDataExternally(X,Y,Z,W,atoms,coreRepresentation)
#     X,Y,Z,W,Xf,Yf,Zf,Wf,pointsPerCell_coarse, pointsPerCell_fine, atoms,PSPs,nPoints,nOrbitals,nElectrons,referenceEigenvalues = buildMeshFromMinimumDepthCells(domainSize,domainSize,domainSize,maxSideLength,coreRepresentation,
#                                                                                                      inputFile,outputFile,srcdir,order,fine_order,gaugeShift,
#                                                                                                      divideCriterion,divideParameter1,divideParameter2,divideParameter3,divideParameter4)
    
    # compute aspect ratios
    
    xl = np.max(X) - np.min(X)
    yl = np.max(Y) - np.min(Y)
    zl = np.max(Z) - np.min(Z)
    
    aspectRatio = max( xl, max(yl,zl) )/ min( xl, min(yl,zl) )
    
    
    # Gather all the remote points onto rank 0
    if rank==0:
        RANKS=np.zeros(len(X),dtype=np.int)
        rprint(rank,"Initial rank 0 has %i points." %len(X))
    for sender in range(1,size):
        rprint(rank,"Getting data from rank %i" %sender)
        if rank==0:
            numData = comm.recv(source=sender)
            rprint(rank,"Number of remote points: %i" %numData)
            tempX = np.empty(numData)
            tempY = np.empty(numData)
            tempZ = np.empty(numData)
            tempRHO = np.empty(numData)
            tempW = np.empty(numData)
            comm.Recv(tempX, source=sender)
            comm.Recv(tempY, source=sender)
            comm.Recv(tempZ, source=sender)
            comm.Recv(tempRHO, source=sender)
            comm.Recv(tempW, source=sender)
            
            X = np.append(X,tempX)
            Y = np.append(Y,tempY)
            Z = np.append(Z,tempZ)
            RHO = np.append(RHO,tempRHO)
            W = np.append(W,tempW)
            RANKS = np.append(RANKS, sender*np.ones(numData,dtype=np.int))
            rprint(rank,"New length of X: %i" %len(X))
        elif rank==sender:
            comm.send(len(X), dest=0)
            comm.Send(X,dest=0)
            comm.Send(Y,dest=0)
            comm.Send(Z,dest=0)
            comm.Send(RHO,dest=0)
            comm.Send(W,dest=0)
        else:
            pass
        
    if rank==0:
        np.savetxt(outputFile+'-X.csv', X, delimiter=',')
        np.savetxt(outputFile+'-Y.csv', Y, delimiter=',')
        np.savetxt(outputFile+'-Z.csv', Z, delimiter=',')
        np.savetxt(outputFile+'-RHO.csv', RHO, delimiter=',')
        np.savetxt(outputFile+'-W.csv', W, delimiter=',')
        np.savetxt(outputFile+'-RANKS.csv', RANKS, delimiter=',')
        
        
    comm.barrier()
    rprint(0,"rank %i aspect ratio = %f" %(rank,aspectRatio))
    
    
    
    
    
#     ## FOR VISIT OR PARAVIEW CONNECTIVITY GRAPHS NEED TO UNCOMMENT STUFF BELOW AND EXTRACT TREE FROM ABOVE
#     X,Y,Z,W,RHO, XV, YV, ZV, vertexIdx, centerIdx, ghostCells = tree.extractXYZ_connected()
#     
# #     print("Saving mesh to /home/njvaughn/PSPmesh/")
# #     np.save("/home/njvaughn/PSPmesh/X_%i" %(nPoints), X)
# #     np.save("/home/njvaughn/PSPmesh/Y_%i" %(nPoints), Y)
# #     np.save("/home/njvaughn/PSPmesh/Z_%i" %(nPoints), Z)
# #     np.save("/home/njvaughn/PSPmesh/W_%i" %(nPoints), W)
# #     np.save("/home/njvaughn/PSPmesh/RHO_%i" %(nPoints), RHO)
#     
# #     print(XV)
# #     print(YV)
# #     print(ZV)
# #     print(len(XV))
# #     print(XV.size)
# #     print(RHO)
# #     print(vertexIdx)
# #     print(centerIdx)
# 
#     print("Number of coarse mesh points = %i" %len(X))
#     print("Number of coarse mesh cells  = %i" %len(pointsPerCell_coarse))
#     print("Number of fine mesh points   = %i" %len(Xf))
# 
#     conn=np.zeros(XV.size)
#     for i in range(len(conn)):
#         conn[i] = i
#     offset=np.zeros(int(XV.size/8))
#     for i in range(len(offset)):
#         offset[i] = 8*(i+1)
#     ctype = np.zeros(len(offset))
#     for i in range(len(ctype)):
#         ctype[i] = VtkVoxel.tid
#     pointVals = {"density":np.zeros(XV.size)}
#     x1=y1=z1=-1
#     x2=y2=z2=1
#     for i in range(len(XV)):
#         pointVals["density"][i] = max( RHO[vertexIdx[i]], 1e-16) 
# #         r1 = np.sqrt(  (XV[i]-x1)*(XV[i]-x1) + (YV[i]-y1)*(YV[i]-y1) + (ZV[i]-z1)*(ZV[i]-z1) )
# #         r2 = np.sqrt(  (XV[i]-x2)*(XV[i]-x2) + (YV[i]-y2)*(YV[i]-y2) + (ZV[i]-z2)*(ZV[i]-z2) )
# # #         print(r)
# #         pointVals["density"][i] = np.exp( - r1) + np.exp( - 2*r2)
# #         pointVals["density_p"][i] = np.exp( - r1 )
#     
#     cellVals = {"density":np.zeros(offset.size)}
# #     for i in range(len(offset)):
# #         
# # #         startIdx = 8*i
# # #         xmid = (XV[startIdx] + XV[startIdx+1])/2
# # #         ymid = (YV[startIdx] + YV[startIdx+2])/2
# # #         zmid = (ZV[startIdx] + ZV[startIdx+4])/2
# # #         
# # #         r1 = np.sqrt( (xmid-x1)**2 + (ymid-y1)**2 + (zmid-z1)**2)
# # #         r2 = np.sqrt( (xmid-x2)**2 + (ymid-y2)**2 + (zmid-z2)**2)
# # #         
# #         cellVals["density"][i] = max( RHO[centerIdx[i]], 1e-16) 
#         
#         
#     
# #     savefile="/Users/nathanvaughn/Desktop/meshTests/forVisitTesting/beryllium"
#     unstructuredGridToVTK(outputFile, 
#                           XV, YV, ZV, connectivity = conn, offsets = offset, cell_types = ctype, 
#                           cellData = cellVals, pointData = pointVals)
# #     x=[]
# #     y=[]
# #     z=[]
# #     w=[]
# #     rho=[]
# #     for i in range(len(X)):
# #         if ( (Z[i]>=-300.3) and (Z[i]<300.3) ):
# #             newPoint=True
# # #             for j in range(min(len(x),100)):
# # #                 if ( (X[i] == x[-j]) and (Y[i] == y[-j]) ): 
# # #                     newPoint=False
# # #                     print('This (x,y) has already been added.')
# #             if newPoint==True: 
# #                 
# #                 x.append(X[i])
# #                 y.append(Y[i])
# # #                 z.append(Z[i])
# #                 z.append(0)
# #                 w.append(W[i])
# # #                 rho.append(RHO[i])
# #                 rho.append(np.exp(-(np.sqrt(X[i]**2 + Y[i]**2 + Z[i]**2))))
# #     print('Number of plotting points: ', len(x))
# #     
# #     print('About to export mesh.')
# #     pointsToVTK(outputFile, np.array(x), np.array(y), np.array(z), data = 
# #                     {"rho" : np.array(rho)} )
# # #     tree.exportGridpoints(outputFile)
# # #     tree.orthonormalizeOrbitals()
# # #     tree.exportGridpoints('/Users/nathanvaughn/Desktop/meshTests/CO_afterOrth')
# 
# 
#     print('Meshes Exported.')
    
    
    tree=[]
    return tree


# def timeConvolutions(domainSize,maxSideLength,coreRepresentation, 
#                           inputFile,outputFile,srcdir,order,gaugeShift,
#                           divideCriterion,divideParameter1,divideParameter2,divideParameter3,divideParameter4):    
#     
#     
# #     [coordinateFile, DummyOutputFile] = np.genfromtxt(inputFile,dtype="|U100")[:2]
#     [coordinateFile, referenceEigenvaluesFile, DummyOutputFile] = np.genfromtxt(inputFile,dtype="|U100")[:3]
#     [Eband, Ekinetic, Eexchange, Ecorrelation, Eelectrostatic, Etotal] = np.genfromtxt(inputFile)[3:]
# 
#     print([coordinateFile, Etotal, Eexchange, Ecorrelation, Eband])
# 
#     
#     X,Y,Z,W,Xf,Yf,Zf,Wf,pointsPerCell_coarse, pointsPerCell_fine, atoms,PSPs,nPoints,nOrbitals,nElectrons,referenceEigenvalues,tree = buildMeshFromMinimumDepthCells(domainSize,domainSize,domainSize,maxSideLength,coreRepresentation,
#                                                                                                      inputFile,outputFile,srcdir,order,order,gaugeShift,
#                                                                                                      MESHTYPE,MESHPARAM1,MESHPARAM2,MESHPARAM3,MESHPARAM4,saveTree=True)
#    
#     print("Number of points: ", len(X))
# 
#     return


def plotMeshPoints(outputFile):
    
    X=np.loadtxt(outputFile+'-X.csv', delimiter=',')
    Y=np.loadtxt(outputFile+'-Y.csv', delimiter=',')
    Z=np.loadtxt(outputFile+'-Z.csv', delimiter=',')
    RHO=np.loadtxt(outputFile+'-RHO.csv', delimiter=',')
    RANKS=np.loadtxt(outputFile+'-RANKS.csv', delimiter=',')


    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    
    ax.scatter(X, Y, Z, c = RANKS)
#     plt.savefig(outputFile+'_'+str(size)+"_rank_decomposition.png")
#     plt.show()
    
    
    for ii in range(180):
        rprint(rank,2*ii)
        ax.view_init(elev=10., azim=2*ii)
        plt.savefig(outputFile+'_'+str(size)+"_rank_decomposition%d.png" % ii)
        
        

def time_convolutions(  coordinates, kernelName,  approximationName, outputFile):
    
    ## Load in data.
    X=np.loadtxt(coordinates+'-X.csv', delimiter=',')
    Y=np.loadtxt(coordinates+'-Y.csv', delimiter=',')
    Z=np.loadtxt(coordinates+'-Z.csv', delimiter=',')
    RHO=np.loadtxt(coordinates+'-RHO.csv', delimiter=',')
    W=np.loadtxt(coordinates+'-W.csv', delimiter=',')
    nPoints=len(X)
    
    
    # Treecode parameters
    GPUpresent=True
    treecode_verbosity=1
    maxParNode=4000
    batchSize=4000
    singularity=BT.Singularity.SUBTRACTION
    computeType=BT.ComputeType.PARTICLE_CLUSTER
    
    
    
    if kernelName=="coulomb":
        kernel=BT.Kernel.COULOMB
    elif kernelName=="yukawa": 
        kernel=BT.Kernel.YUKAWA
    else:
        print("What should kernelName be?")
         
    numberOfKernelParameters=1 
    kernelParameters=np.array([1.0])
    
    
    
    if approximationName=="lagrange":
        approximation=BT.Approximation.LAGRANGE
    elif approximationName=="hermite":
        approximation=BT.Approximation.HERMITE
    
    
    
#     print("nPoints = ", nPoints)
#     exit(-1)
    direct = BT.callTreedriver(     nPoints, nPoints, 
                                    np.copy(X), np.copy(Y), np.copy(Z), np.copy(RHO), 
                                    np.copy(X), np.copy(Y), np.copy(Z), np.copy(RHO), np.copy(W),
                                    kernel, numberOfKernelParameters, kernelParameters, 
                                    singularity, approximation, computeType,
                                    1, 0.0, maxParNode, batchSize,
                                    GPUpresent, treecode_verbosity
                                    )


    treecode_verbosity=0
    
    for theta in [0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85, 0.9]:
        for treecodeOrder in [2,3,4,5,6,7,8]:
            start=time.time()
            tree = BT.callTreedriver(   nPoints, nPoints, 
                                        np.copy(X), np.copy(Y), np.copy(Z), np.copy(RHO), 
                                        np.copy(X), np.copy(Y), np.copy(Z), np.copy(RHO), np.copy(W),
                                        kernel, numberOfKernelParameters, kernelParameters, 
                                        singularity, approximation, computeType,
                                        treecodeOrder, theta, maxParNode, batchSize,
                                        GPUpresent, treecode_verbosity
                                        )
            end=time.time()
            runtime=end-start
            # measure error
            
            L2err = np.sqrt( np.dot( (tree-direct)**2, W) ) / np.sqrt( np.dot( (direct)**2, W) )
            rprint(rank,"theta = %1.1f, order = %i,     error = %1.2e,     time = %1.3f" %(theta,treecodeOrder,L2err,runtime))
            
            
            # write to data file
            header = ["kernel", "approximation", "meshSize", "order", "theta", "batchSize", "clusterSize", "error", "time"]
            myData = [kernelName, approximationName, nPoints, treecodeOrder, theta, batchSize, maxParNode, L2err, runtime]
            if not os.path.isfile(outputFile):
                myFile = open(outputFile, 'a')
                with myFile:
                    writer = csv.writer(myFile)
                    writer.writerow(header) 
                
            
            myFile = open(outputFile, 'a')
            with myFile:
                writer = csv.writer(myFile)
                writer.writerow(myData)
    
    




if __name__ == "__main__":
    gaugeShift=-0.5
    
    
#     import matplotlib
#     matplotlib.use('Qt4Agg')
#     from matplotlib import pyplot as plt
#     from mpl_toolkits.mplot3d import Axes3D
    rprint(rank, "matplotlib.get_backend(): ", matplotlib.get_backend())

    
# #     ## THIS WAS USED TO GENERATE FIGURES IN PAPER
#     tree = exportMeshForParaview(domain=30,order=5,
#                         minDepth=3, maxDepth=20, additionalDepthAtAtoms=0, divideCriterion='ParentChildrenIntegral', 
#                         divideParameter1=1500, divideParameter2=0, divideParameter3=1e-7, divideParameter4=4,
#                         smoothingEpsilon=0.0,inputFile='../src/utilities/molecularConfigurations/benzeneAuxiliary.csv', 
#                         outputFile='/Users/nathanvaughn/Desktop/meshTests/forPaper/benzene_1e-7_rotated2D',
#                         savedMesh='') 



#     # Set domain to be an integrer number of far-field cells
#     n=1
#     domainSize          = int(sys.argv[n]); n+=1
#     maxSideLength       = float(sys.argv[n]); n+=1
#  
#     print("original domain size: ", domainSize)
#     remainder=(2*domainSize)%maxSideLength
#     print("remainder: ", remainder)
#     if remainder>0:
#         domainSize+=1/2*(maxSideLength-remainder)
#     print("Max side length: ",maxSideLength)
#     print("Domain length after adjustment: ", domainSize)
#     print(" Far field nx, ny, nz = ", 2*domainSize/maxSideLength)



    
#     inputFile=srcdir+'molecularConfigurations/SiliconClusterAuxiliaryPSP.csv'

#     inputFile=srcdir+'molecularConfigurations/siliconAuxiliaryPSP.csv'
#     outputFile="/Users/nathanvaughn/Desktop/meshTests/PSPmeshes/silicon-PSP"
    
#     inputFile=srcdir+'molecularConfigurations/Si2AuxiliaryPSP.csv'
# #     outputFile="/Users/nathanvaughn/Desktop/meshTests/PSPmeshes/Si2"
#     outputFile="/home/njvaughn/PSPmesh/Si2-PSP"

#     inputFile=srcdir+'molecularConfigurations/C20AuxiliaryPSP.csv'
#     outputFile="/Users/nathanvaughn/Desktop/meshTests/PSPmeshes/C20"
#     outputFile="/home/njvaughn/PSPmesh/C20"

    inputFile=srcdir+'molecularConfigurations/C60AuxiliaryPSP.csv'
#     outputFile="/Users/nathanvaughn/Desktop/meshTests/PSPmeshes/C60-PSP"
    outputFile="/home/njvaughn/PSPmesh/C60-PSP"
    
    
    coreRepresentation="Pseudopotential"
    MESHTYPE='coarsenedUniformTwoLevel'
 
    order=3
    gaugeShift=-0.5
     
    domainSize=64
    MAXSIDELENGTH=8
    
    MESHPARAM1=0.5 # near field spacing 
    MESHPARAM2=8.0 # far field spacing
    MESHPARAM3=2.0 # ball radius
    MESHPARAM4=0 # additional inner refinement 
    
     
    
#     coreRepresentation="AllElectron"
#     MESHTYPE='ParentChildrenIntegral'
#     order=4
#     MESHPARAM1=1e-5
#     MESHPARAM2=1e10
#     MESHPARAM3=1e10
#     MESHPARAM4=1e10
    
#     
# #     inputFile=srcdir+'molecularConfigurations/berylliumAuxiliary.csv'
#     inputFile=srcdir+'molecularConfigurations/siliconAuxiliary.csv'
#     outputFile="/Users/nathanvaughn/Desktop/meshTests/PSPmeshes/silicon-AE"
    
    
#     domainSize=16
#     MAXSIDELENGTH=32
#     MESHPARAM1=0.5 # near field spacing 
#     MESHPARAM2=8.0 # far field spacing
#     MESHPARAM3=2.0 # ball radius
#     MESHPARAM4=0 # additional inner refinement  
    
#     MESHTYPE="ParentChildrenIntegral"
#     MESHPARAM1=5e-8
#     
#    MAXSIDELENGTH=16
#     MESHTYPE="VPsiIntegral"
#    MESHTYPE="ChiIntegral"
#     MESHPARAM1=1e-3
#     order=4
    

  
#     tree = exportMeshForParaview(domainSize,MAXSIDELENGTH,coreRepresentation, 
#                           inputFile,outputFile,srcdir,order,gaugeShift,
#                           MESHTYPE,MESHPARAM1,MESHPARAM2,MESHPARAM3,MESHPARAM4)

    time_convolutions(  outputFile, "coulomb",  "lagrange", "/home/njvaughn/GLsync/hermite_vs_lagrange/C60/coulomb-times.csv")
    time_convolutions(  outputFile, "yukawa",  "lagrange", "/home/njvaughn/GLsync/hermite_vs_lagrange/C60/yukawa-times.csv")
#     time_convolutions(  outputFile, "yukawa",  "lagrange", "/home/njvaughn/GLsync/hermite_vs_lagrange/Si2/dummy-times.csv")
#     time_convolutions(  outputFile, "yukawa",  "hermite", "/home/njvaughn/GLsync/hermite_vs_lagrange/Si2/yukawa-times.csv")
    
#     if rank==0:
#         plotMeshPoints(outputFile)
#         
#     
#     tree = timeConvolutions(domainSize,MAXSIDELENGTH,coreRepresentation, 
#                           inputFile,outputFile,srcdir,order,gaugeShift,
#                           MESHTYPE,MESHPARAM1,MESHPARAM2,MESHPARAM3,MESHPARAM4)
     
    
    
    
