'''
Created on Jun 25, 2018

@author: nathanvaughn
'''
import sys
sys.path.append('../src/dataStructures')
sys.path.append('../src/utilities')
import itertools
import time
import numpy as np
# import dask.array as da
import matplotlib.pyplot as plt
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


from meshUtilities import *
ThreeByThreeByThree = [element for element in itertools.product(range(3),range(3),range(3))]

from TreeStruct_CC import Tree


def find(a, x):
    'Locate the leftmost value exactly equal to x'
    i = bisect.bisect_left(a, [x,])
    if i != len(a) and a[i][0] == x:
        return i
    raise ValueError




def exportMeshForParaview(domain,order,minDepth, maxDepth, additionalDepthAtAtoms, divideCriterion, 
                          divideParameter1, divideParameter2=0.0, divideParameter3=0.0, divideParameter4=0.0, 
                          smoothingEpsilon=0.0, 
                          inputFile='', outputFile='',
                          savedMesh=''):    
    
    
#     [coordinateFile, DummyOutputFile] = np.genfromtxt(inputFile,dtype="|U100")[:2]
    [coordinateFile, referenceEigenvaluesFile, DummyOutputFile] = np.genfromtxt(inputFile,dtype="|U100")[:3]
    [Eband, Ekinetic, Eexchange, Ecorrelation, Eelectrostatic, Etotal] = np.genfromtxt(inputFile)[3:]
#     nElectrons = int(nElectrons)
#     nOrbitals = int(nOrbitals)

    print('Reading atomic coordinates from: ', coordinateFile)
    atomData = np.genfromtxt(coordinateFile,delimiter=',',dtype=float)
    if np.shape(atomData)==(5,):
        nElectrons = atomData[3]
    else:
        nElectrons = 0
        for i in range(len(atomData)):
            nElectrons += atomData[i,3]
    
#     nOrbitals = int( np.ceil(nElectrons/2))
    nOrbitals = int( np.ceil(nElectrons/2)+1)

    if inputFile=='../src/utilities/molecularConfigurations/benzeneAuxiliary.csv':
        nOrbitals = 30
        
    if inputFile=='../src/utilities/molecularConfigurations/O2Auxiliary.csv':
        nOrbitals = 10
        
    print('nElectrons = ', nElectrons)
    print('nOrbitals  = ', nOrbitals)
    print([coordinateFile, Etotal, Eexchange, Ecorrelation, Eband])
    tree = Tree(-domain,domain,order,-domain,domain,order,-domain,domain,order,nElectrons,nOrbitals,additionalDepthAtAtoms=additionalDepthAtAtoms,minDepth=minDepth,gaugeShift=gaugeShift,
                coordinateFile=coordinateFile,smoothingEps=smoothingEpsilon,inputFile=inputFile)#, iterationOutFile=outputFile)

    
    print('max depth ', maxDepth)
#     tree.minimalBuildTree( maxLevels=maxDepth, initializationType='atomic',divideCriterion=divideCriterion, 
    tree.buildTree( maxLevels=maxDepth, initializationType='atomic',divideCriterion=divideCriterion, 
                    divideParameter1=divideParameter1, divideParameter2=divideParameter2, divideParameter3=divideParameter3, divideParameter4=divideParameter4, 
                    savedMesh=savedMesh, printTreeProperties=True,onlyFillOne=False)
    
    X,Y,Z,W,RHO, XV, YV, ZV, vertexIdx, centerIdx, ghostCells = tree.extractXYZ_connected()

    print(XV)
    print(YV)
    print(ZV)
    print(len(XV))
    print(XV.size)
    print(RHO)
    print(vertexIdx)
    print(centerIdx)
    conn=np.zeros(XV.size)
    for i in range(len(conn)):
        conn[i] = i
    offset=np.zeros(int(XV.size/8))
    for i in range(len(offset)):
        offset[i] = 8*(i+1)
    ctype = np.zeros(len(offset))
    for i in range(len(ctype)):
        ctype[i] = VtkVoxel.tid
    pointVals = {"density":np.zeros(XV.size)}
    x1=y1=z1=-1
    x2=y2=z2=1
    for i in range(len(XV)):
        pointVals["density"][i] = max( RHO[vertexIdx[i]], 1e-16) 
#         r1 = np.sqrt(  (XV[i]-x1)*(XV[i]-x1) + (YV[i]-y1)*(YV[i]-y1) + (ZV[i]-z1)*(ZV[i]-z1) )
#         r2 = np.sqrt(  (XV[i]-x2)*(XV[i]-x2) + (YV[i]-y2)*(YV[i]-y2) + (ZV[i]-z2)*(ZV[i]-z2) )
# #         print(r)
#         pointVals["density"][i] = np.exp( - r1) + np.exp( - 2*r2)
#         pointVals["density_p"][i] = np.exp( - r1 )
    
    cellVals = {"density":np.zeros(offset.size)}
#     for i in range(len(offset)):
#         
# #         startIdx = 8*i
# #         xmid = (XV[startIdx] + XV[startIdx+1])/2
# #         ymid = (YV[startIdx] + YV[startIdx+2])/2
# #         zmid = (ZV[startIdx] + ZV[startIdx+4])/2
# #         
# #         r1 = np.sqrt( (xmid-x1)**2 + (ymid-y1)**2 + (zmid-z1)**2)
# #         r2 = np.sqrt( (xmid-x2)**2 + (ymid-y2)**2 + (zmid-z2)**2)
# #         
#         cellVals["density"][i] = max( RHO[centerIdx[i]], 1e-16) 
        
        
    
#     savefile="/Users/nathanvaughn/Desktop/meshTests/forVisitTesting/beryllium"
    unstructuredGridToVTK(outputFile, 
                          XV, YV, ZV, connectivity = conn, offsets = offset, cell_types = ctype, 
                          cellData = cellVals, pointData = pointVals)
#     x=[]
#     y=[]
#     z=[]
#     w=[]
#     rho=[]
#     for i in range(len(X)):
#         if ( (Z[i]>=-300.3) and (Z[i]<300.3) ):
#             newPoint=True
# #             for j in range(min(len(x),100)):
# #                 if ( (X[i] == x[-j]) and (Y[i] == y[-j]) ): 
# #                     newPoint=False
# #                     print('This (x,y) has already been added.')
#             if newPoint==True: 
#                 
#                 x.append(X[i])
#                 y.append(Y[i])
# #                 z.append(Z[i])
#                 z.append(0)
#                 w.append(W[i])
# #                 rho.append(RHO[i])
#                 rho.append(np.exp(-(np.sqrt(X[i]**2 + Y[i]**2 + Z[i]**2))))
#     print('Number of plotting points: ', len(x))
#     
#     print('About to export mesh.')
#     pointsToVTK(outputFile, np.array(x), np.array(y), np.array(z), data = 
#                     {"rho" : np.array(rho)} )
# #     tree.exportGridpoints(outputFile)
# #     tree.orthonormalizeOrbitals()
# #     tree.exportGridpoints('/Users/nathanvaughn/Desktop/meshTests/CO_afterOrth')

    print('Meshes Exported.')
    
#     tree.saveList = ['']
#     for _,cell in tree.masterList:
#         if cell.leaf==True:
#             tree.saveList.insert(bisect.bisect_left(cell.tree.saveList, cell.uniqueID), [cell.uniqueID] )
    #     cell.tree.masterList.insert(bisect.bisect_left(cell.tree.masterList, [children[i,j,k].uniqueID,]), [children[i,j,k].uniqueID,children[i,j,k]])

    
    return tree

# def testDask(domain,order,minDepth, maxDepth, additionalDepthAtAtoms, divideCriterion, 
#                           divideParameter1, divideParameter2=0.0, divideParameter3=0.0, divideParameter4=0.0, 
#                           smoothingEpsilon=0.0, 
#                           inputFile='', outputFile='',
#                           savedMesh=''):    
#     
#     
# #     [coordinateFile, DummyOutputFile] = np.genfromtxt(inputFile,dtype="|U100")[:2]
#     [coordinateFile, referenceEigenvaluesFile, DummyOutputFile] = np.genfromtxt(inputFile,dtype="|U100")[:3]
#     [Eband, Ekinetic, Eexchange, Ecorrelation, Eelectrostatic, Etotal] = np.genfromtxt(inputFile)[3:]
# #     nElectrons = int(nElectrons)
# #     nOrbitals = int(nOrbitals)
# 
#     print('Reading atomic coordinates from: ', coordinateFile)
#     atomData = np.genfromtxt(coordinateFile,delimiter=',',dtype=float)
#     if np.shape(atomData)==(5,):
#         nElectrons = atomData[3]
#     else:
#         nElectrons = 0
#         for i in range(len(atomData)):
#             nElectrons += atomData[i,3]
#     
# #     nOrbitals = int( np.ceil(nElectrons/2))
#     nOrbitals = int( np.ceil(nElectrons/2)+1)
# 
#     if inputFile=='../src/utilities/molecularConfigurations/benzeneAuxiliary.csv':
#         nOrbitals = 30
#         
#     if inputFile=='../src/utilities/molecularConfigurations/O2Auxiliary.csv':
#         nOrbitals = 10
#         
#     print('nElectrons = ', nElectrons)
#     print('nOrbitals  = ', nOrbitals)
#     print([coordinateFile, Etotal, Eexchange, Ecorrelation, Eband])
#     tree = Tree(-domain,domain,order,-domain,domain,order,-domain,domain,order,nElectrons,nOrbitals,additionalDepthAtAtoms=additionalDepthAtAtoms,minDepth=minDepth,gaugeShift=gaugeShift,
#                 coordinateFile=coordinateFile,smoothingEps=smoothingEpsilon,inputFile=inputFile)#, iterationOutFile=outputFile)
# 
#     
#     print('max depth ', maxDepth)
#     tree.minimalBuildTree( maxLevels=maxDepth, initializationType='atomic',divideCriterion=divideCriterion, 
#                     divideParameter1=divideParameter1, divideParameter2=divideParameter2, divideParameter3=divideParameter3, divideParameter4=divideParameter4, 
#                     savedMesh=savedMesh, printTreeProperties=True,onlyFillOne=False)
#     
#     x,y,z,w = tree.extractXYZ()
#     return x,y,z,w
# #     from dask.distributed import Client, progress
# #     client = Client(processes=False, threads_per_worker=4,
# #                     n_workers=1, memory_limit='2GB')
# #     client
#     CHUNKSIZE=10000
#     X = da.from_array(x,chunks=(CHUNKSIZE,))
#     Y = da.from_array(y,chunks=(CHUNKSIZE,))
#     Z = da.from_array(z,chunks=(CHUNKSIZE,))
#     W = da.from_array(w,chunks=(CHUNKSIZE,))
#     nPoints = len(x)
#     print(nPoints)
#     print(X)
#     print(type(x))
#     print(type(X))
#     tree=None
#     
#     wavefunctions = da.zeros( (nPoints,nOrbitals), chunks=(CHUNKSIZE,1) )
#     Veff = da.zeros( (nPoints,), chunks=(CHUNKSIZE,))
#     rho = da.zeros( (nPoints,), chunks=(CHUNKSIZE,))
#     
#     
#     return




if __name__ == "__main__":
    gaugeShift=-0.5

    
#     ## THIS WAS USED TO GENERATE FIGURES IN PAPER
#     tree = exportMeshForParaview(domain=30,order=5,
#                         minDepth=3, maxDepth=20, additionalDepthAtAtoms=0, divideCriterion='ParentChildrenIntegral', 
#                         divideParameter1=1500, divideParameter2=0, divideParameter3=1e-7, divideParameter4=4,
#                         smoothingEpsilon=0.0,inputFile='../src/utilities/molecularConfigurations/benzeneAuxiliary.csv', 
#                         outputFile='/Users/nathanvaughn/Desktop/meshTests/forPaper/benzene_1e-7_rotated2D',
#                         savedMesh='')   

