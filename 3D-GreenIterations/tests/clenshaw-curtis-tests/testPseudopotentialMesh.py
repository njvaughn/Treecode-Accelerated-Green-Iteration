'''
Created on Jun 25, 2018

@author: nathanvaughn
'''
import sys

srcdir="/Users/nathanvaughn/Documents/GitHub/TAGI/3D-GreenIterations/src/"
sys.path.append(srcdir+'dataStructures')
sys.path.append(srcdir+'Green-Iteration-Routines')
sys.path.append(srcdir+'utilities')
sys.path.append(srcdir+'../ctypesTests/src')

sys.path.append(srcdir+'../ctypesTests')
sys.path.append(srcdir+'../ctypesTests/lib')

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


sys.path.insert(1, '/Users/nathanvaughn/Documents/GitHub/TAGI/3D-GreenIterations/src/utilities')
sys.path.insert(1, '/Users/nathanvaughn/Documents/GitHub/TAGI/3D-GreenIterations/src/dataStructures')
sys.path.insert(1, '/home/njvaughn/TAGI/3D-GreenIterations/src/utilities')
from loadBalancer import loadBalance
from mpiUtilities import global_dot, scatterArrays, rprint
from mpiMeshBuilding import  buildMeshFromMinimumDepthCells

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
    
    X,Y,Z,W,Xf,Yf,Zf,Wf,atoms,PSPs,nPoints,nOrbitals,nElectrons,referenceEigenvalues,tree = buildMeshFromMinimumDepthCells(domainSize,domainSize,domainSize,maxSideLength,coreRepresentation,
                                                                                                     inputFile,outputFile,srcdir,order,order,gaugeShift,
                                                                                                     MESHTYPE,MESHPARAM1,MESHPARAM2,MESHPARAM3,MESHPARAM4,saveTree=True)
   

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
    

    
    return tree






if __name__ == "__main__":
    gaugeShift=-0.5

    
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

    inputFile=srcdir+'molecularConfigurations/berylliumAuxiliaryPSP.csv'
    outputFile="/Users/nathanvaughn/Desktop/meshTests/PSPmeshes/beryllium"

#     inputFile=srcdir+'molecularConfigurations/C20AuxiliaryPSP.csv'
#     outputFile="/Users/nathanvaughn/Desktop/meshTests/PSPmeshes/C20"
    
    
    coreRepresentation="Pseudopotential"
    MESHTYPE='coarsenedUniform'
    order=4
    gaugeShift=-0.5
    
    domainSize=12.7*2
    MAXSIDELENGTH=25.4*2
    MESHPARAM1=0.4 # near field spacing 
    MESHPARAM2=6.4 # far field spacing
    MESHPARAM3=1.6 # ball radius
    MESHPARAM4=0 # additional inner refinement 
    
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
    

  
    tree = exportMeshForParaview(domainSize,MAXSIDELENGTH,coreRepresentation, 
                          inputFile,outputFile,srcdir,order,gaugeShift,
                          MESHTYPE,MESHPARAM1,MESHPARAM2,MESHPARAM3,MESHPARAM4)
     
    
    
    
