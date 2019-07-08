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
import dask.array as da
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

def exportMeshForTreecodeTesting(domain,order,minDepth, maxDepth, additionalDepthAtAtoms, divideCriterion, divideParameter1, divideParameter2, divideParameter3, divideParameter4, inputFile,
                                 smoothingEpsilon=0.0,
                                 savedMesh=''):

    [coordinateFile, referenceEigenvaluesFile, DummyOutputFile] = np.genfromtxt(inputFile,dtype="|U100")[:3]
    [Eband, Ekinetic, Eexchange, Ecorrelation, Eelectrostatic, Etotal] = np.genfromtxt(inputFile)[3:9]


    print('Reading atomic coordinates from: ', coordinateFile)
    atomData = np.genfromtxt(coordinateFile,delimiter=',',dtype=float)
    if np.shape(atomData)==(5,):
        nElectrons = atomData[3]
    else:
        nElectrons = 0
        for i in range(len(atomData)):
            nElectrons += atomData[i,3]
    
    nOrbitals = int( np.ceil(nElectrons/2)+1)
    print('nElectrons = ', nElectrons)
    print('nOrbitals  = ', nOrbitals)
    print([coordinateFile, Etotal, Eexchange, Ecorrelation, Eband])
    tree = Tree(-domain,domain,order,-domain,domain,order,-domain,domain,order,nElectrons,nOrbitals,additionalDepthAtAtoms=additionalDepthAtAtoms,minDepth=minDepth,gaugeShift=gaugeShift,
                coordinateFile=coordinateFile,smoothingEps=smoothingEpsilon,inputFile=inputFile)#, iterationOutFile=outputFile)

    
    print('max depth ', maxDepth)
    tree.buildTree( maxLevels=maxDepth, initializationType='atomic',divideCriterion=divideCriterion, 
                    divideParameter1=divideParameter1, divideParameter2=divideParameter2, divideParameter3=divideParameter3, divideParameter4=divideParameter4, 
                    savedMesh=savedMesh, printTreeProperties=True,onlyFillOne=False)
    
#     sourcesTXT = '../examples/S%ipy.txt' %tree.numberOfGridpoints
#     targetsTXT = '../examples/T%ipy.txt' %tree.numberOfGridpoints
    
#     sourcesTXT = '/Users/nathanvaughn/Documents/GitHub/hybrid-gpu-treecode/examplesOxygenAtom/S%ipy.txt' %tree.numberOfGridpoints
    sourcesTXT = '/Users/nathanvaughn/Desktop/S%i.txt' %tree.numberOfGridpoints
    targetsTXT = '/Users/nathanvaughn/Desktop/T%i.txt' %tree.numberOfGridpoints
    
    Sources = tree.extractLeavesDensity() 
    Targets = tree.extractLeavesDensity()

#     print(Targets[0,:])
    print(Targets[0:3,0:4])
    
    # Save as .txt files
    np.savetxt(sourcesTXT, Sources)
    np.savetxt(targetsTXT, Targets[:,0:4])

    print('Meshes Exported.')   
    



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
    tree.minimalBuildTree( maxLevels=maxDepth, initializationType='atomic',divideCriterion=divideCriterion, 
                    divideParameter1=divideParameter1, divideParameter2=divideParameter2, divideParameter3=divideParameter3, divideParameter4=divideParameter4, 
                    savedMesh=savedMesh, printTreeProperties=True,onlyFillOne=False)
    
    X,Y,Z,W,RHO, XV, YV, ZV, quadIdx = tree.extractXYZ()

    print(XV)
    print(YV)
    print(ZV)
    print(len(XV))
    print(XV.size)
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
        r1 = np.sqrt(  (XV[i]-x1)*(XV[i]-x1) + (YV[i]-y1)*(YV[i]-y1) + (ZV[i]-z1)*(ZV[i]-z1) )
        r2 = np.sqrt(  (XV[i]-x2)*(XV[i]-x2) + (YV[i]-y2)*(YV[i]-y2) + (ZV[i]-z2)*(ZV[i]-z2) )
#         print(r)
        pointVals["density"][i] = np.exp( - r1) + np.exp( - 2*r2)
    
#     savefile="/Users/nathanvaughn/Desktop/meshTests/forVisitTesting/beryllium"
    unstructuredGridToVTK(outputFile, 
                          XV, YV, ZV, connectivity = conn, offsets = offset, cell_types = ctype, 
                          cellData = None, pointData = pointVals)
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

def testDask(domain,order,minDepth, maxDepth, additionalDepthAtAtoms, divideCriterion, 
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
    tree.minimalBuildTree( maxLevels=maxDepth, initializationType='atomic',divideCriterion=divideCriterion, 
                    divideParameter1=divideParameter1, divideParameter2=divideParameter2, divideParameter3=divideParameter3, divideParameter4=divideParameter4, 
                    savedMesh=savedMesh, printTreeProperties=True,onlyFillOne=False)
    
    x,y,z,w = tree.extractXYZ()
    return x,y,z,w
#     from dask.distributed import Client, progress
#     client = Client(processes=False, threads_per_worker=4,
#                     n_workers=1, memory_limit='2GB')
#     client
    CHUNKSIZE=10000
    X = da.from_array(x,chunks=(CHUNKSIZE,))
    Y = da.from_array(y,chunks=(CHUNKSIZE,))
    Z = da.from_array(z,chunks=(CHUNKSIZE,))
    W = da.from_array(w,chunks=(CHUNKSIZE,))
    nPoints = len(x)
    print(nPoints)
    print(X)
    print(type(x))
    print(type(X))
    tree=None
    
    wavefunctions = da.zeros( (nPoints,nOrbitals), chunks=(CHUNKSIZE,1) )
    Veff = da.zeros( (nPoints,), chunks=(CHUNKSIZE,))
    rho = da.zeros( (nPoints,), chunks=(CHUNKSIZE,))
    
    
    return


def exportMeshToCompareDensity(domain,order,minDepth, maxDepth, additionalDepthAtAtoms, divideCriterion, 
                          divideParameter1, divideParameter2=0.0, divideParameter3=0.0, divideParameter4=0.0, 
                          smoothingEpsilon=0.0, 
                          inputFile='', outputFile='',
                          savedMesh='', restartFilesDir='',iterationNumber=5):    
    
    
    
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
    tree.buildTree( maxLevels=maxDepth, initializationType='atomic',divideCriterion=divideCriterion, 
                    divideParameter1=divideParameter1, divideParameter2=divideParameter2, divideParameter3=divideParameter3, divideParameter4=divideParameter4, 
                    savedMesh=savedMesh, printTreeProperties=True,onlyFillOne=False)
    
    print(tree.levelCounts)
    print()
    print(tree.levelCounts.keys())
    print()
    print(tree.levelCounts.values())
    greenIterationOutFile = outputFile[:-4]+'_GREEN_'+str(tree.numberOfGridpoints)+outputFile[-4:]
    SCFiterationOutFile =   outputFile[:-4]+'_SCF_'+str(tree.numberOfGridpoints)+outputFile[-4:]
    densityPlotsDir =       outputFile[:-4]+'_SCF_'+str(tree.numberOfGridpoints)+'_plots'
#     restartFilesDir =       '/home/njvaughn/restartFiles/'+'restartFiles_'+str(tree.numberOfGridpoints)
    restartFilesDir =       '/Users/nathanvaughn/Documents/synchronizedDataFiles/restartFiles_1416000_after25'

    wavefunctionFile =      restartFilesDir+'/wavefunctions'
    densityFile =           restartFilesDir+'/density'
    inputDensityFile =      restartFilesDir+'/inputdensity'
    outputDensityFile =     restartFilesDir+'/outputdensity'
    vHartreeFile =          restartFilesDir+'/vHartree'
    auxiliaryFile =         restartFilesDir+'/auxiliary'
    
    print()
    print()
    
    
    orbitals = np.load(wavefunctionFile+'.npy')
    oldOrbitals = np.copy(orbitals)
#     for m in range(nOrbitals): 
#         tree.importPhiOnLeaves(orbitals[:,m], m)
    density = np.load(densityFile+'.npy')
#     tree.importDensityOnLeaves(density)
    
    inputDensities = np.load(inputDensityFile+'.npy')
    outputDensities = np.load(outputDensityFile+'.npy')
    
    V_hartreeNew = np.load(vHartreeFile+'.npy')
#     tree.importVhartreeOnLeaves(V_hartreeNew)
#     tree.updateVxcAndVeffAtQuadpoints()
    
    
    # make and save dictionary
    auxiliaryRestartData = np.load(auxiliaryFile+'.npy').item()
    print('type of aux: ', type(auxiliaryRestartData))
    SCFcount = auxiliaryRestartData['SCFcount']
    tree.totalIterationCount = auxiliaryRestartData['totalIterationCount']
    tree.orbitalEnergies = auxiliaryRestartData['eigenvalues'] 
    Eold = auxiliaryRestartData['Eold']
    
    
    targets = tree.extractPhi(0)
    x = targets[:,0]
    y = targets[:,1]
    z = targets[:,2]
    for iterationNumber in range(100):
        outputFile='/Users/nathanvaughn/Desktop/meshTests/benzene/comparingIOofIteration%i'%(iterationNumber)
        pointsToVTK(outputFile, np.array(x), np.array(y), np.array(z), data = 
                        {"rhoResidual"+str(iterationNumber) : np.abs( outputDensities[:,iterationNumber]-inputDensities[:,iterationNumber] )  } )
    
    
#     tree.exportGridpoints(outputFile)

    print('Meshes Exported.')
    

    return tree

def testTreeSaveAndReconstruction(domain,order,minDepth, maxDepth, additionalDepthAtAtoms, divideCriterion, 
                          divideParameter1, divideParameter2=0.0, divideParameter3=0.0, divideParameter4=0.0, 
                          smoothingEpsilon=0.0, 
                          inputFile='', outputFile=''):    
    
    
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
    print('nElectrons = ', nElectrons)
    print('nOrbitals  = ', nOrbitals)
    print([coordinateFile, Etotal, Eexchange, Ecorrelation, Eband])
    tree = Tree(-domain,domain,order,-domain,domain,order,-domain,domain,order,nElectrons,nOrbitals,additionalDepthAtAtoms=additionalDepthAtAtoms,minDepth=minDepth,gaugeShift=gaugeShift,
                coordinateFile=coordinateFile,smoothingEps=smoothingEpsilon,inputFile=inputFile)#, iterationOutFile=outputFile)

    
    print('max depth ', maxDepth)
    tree.buildTree( maxLevels=maxDepth, initializationType='atomic',divideCriterion=divideCriterion, 
                    divideParameter1=divideParameter1, divideParameter2=divideParameter2, divideParameter3=divideParameter3, divideParameter4=divideParameter4, 
                    printTreeProperties=True,onlyFillOne=False)
    
    print(tree.levelCounts)
    print()
    print(tree.levelCounts.keys())
    print()
    print(tree.levelCounts.values())
    

    print('Meshes Exported.')
    
#     tree.saveList = ['']
#     for _,cell in tree.masterList:
#         if cell.leaf==True:
#             tree.saveList.insert(bisect.bisect_left(cell.tree.saveList, cell.uniqueID), cell.uniqueID )
#     #     cell.tree.masterList.insert(bisect.bisect_left(cell.tree.masterList, [children[i,j,k].uniqueID,]), [children[i,j,k].uniqueID,children[i,j,k]])

    
    tree2 = Tree(-domain,domain,order,-domain,domain,order,-domain,domain,order,nElectrons,nOrbitals,additionalDepthAtAtoms=additionalDepthAtAtoms,minDepth=minDepth,gaugeShift=gaugeShift,
                coordinateFile=coordinateFile,smoothingEps=smoothingEpsilon,inputFile=inputFile)#, iterationOutFile=outputFile)

    
    tree2.buildTree( maxLevels=maxDepth, initializationType='atomic',divideCriterion=divideCriterion, 
                    divideParameter1=divideParameter1, divideParameter2=divideParameter2, divideParameter3=divideParameter3, divideParameter4=divideParameter4, 
                    savedMesh='LW5_100_1000000.0_0.001_0.npy', printTreeProperties=True,onlyFillOne=False)
    
    
    return tree, tree2
    
def timingTestsForOrbitalInitializations(domain,order,minDepth, maxDepth, depthAtAtoms, divideCriterion, divideParameter,inputFile):
    [coordinateFile, DummyOutputFile] = np.genfromtxt(inputFile,dtype="|U100")[:2]
    [nElectrons, nOrbitals, Eband, Ekinetic, Eexchange, Ecorrelation, Eelectrostatic, Etotal] = np.genfromtxt(inputFile)[2:]
    nElectrons = int(nElectrons)
    nOrbitals = int(nOrbitals)
    
    print([coordinateFile, nElectrons, nOrbitals, 
     Etotal, Eexchange, Ecorrelation, Eband])
    tree = Tree(-domain,domain,order,-domain,domain,order,-domain,domain,order,nElectrons,nOrbitals,maxDepthAtAtoms=maxDepth,gaugeShift=gaugeShift,
                coordinateFile=coordinateFile,inputFile=inputFile)#, iterationOutFile=outputFile)

    
    print('max depth ', maxDepth)
    tree.buildTree( minLevels=minDepth, maxLevels=maxDepth, initializationType='atomic',divideCriterion=divideCriterion, divideParameter=divideParameter, printTreeProperties=True,onlyFillOne=False)
    
#     afterInternal = tree.extractLeavesDensity()
#     print('Max density = ', max(afterInternal[:,3]))
#     tree.initializeDensityFromAtomicDataExternally()
#     afterExternal = tree.extractLeavesDensity()
#     print('Max diff between internal and external: ', np.max( np.abs(afterInternal[:,3] - afterExternal[:,3] )))



    afterInternal0 = tree.extractPhi(0)
    afterInternal2 = tree.extractPhi(2)
    
    tree.initializeOrbitalsFromAtomicDataExternally()
    
    afterExternal0 = tree.extractPhi(0)
    afterExternal2 = tree.extractPhi(2)
    
    print('Max diff between internal0 and external0: ', np.max( np.abs(afterInternal0[:,3] - afterExternal0[:,3] )))
    print('Max diff between internal2 and external2: ', np.max( np.abs(afterInternal2[:,3] - afterExternal2[:,3] )))
    

def timingTestsForOrbitalOrthogonalizations(domain,order,minDepth, maxDepth, divideCriterion, divideParameter,inputFile):
    [coordinateFile, DummyOutputFile] = np.genfromtxt(inputFile,dtype="|U100")[:2]
    [nElectrons, nOrbitals, Eband, Ekinetic, Eexchange, Ecorrelation, Eelectrostatic, Etotal] = np.genfromtxt(inputFile)[2:]
    nElectrons = int(nElectrons)
    nOrbitals = int(nOrbitals)
    
    print([coordinateFile, nElectrons, nOrbitals, 
     Etotal, Eexchange, Ecorrelation, Eband])
    tree = Tree(-domain,domain,order,-domain,domain,order,-domain,domain,order,nElectrons,nOrbitals,maxDepthAtAtoms=maxDepth,gaugeShift=gaugeShift,
                coordinateFile=coordinateFile,inputFile=inputFile)#, iterationOutFile=outputFile)

    
    tree.buildTree( minLevels=minDepth, maxLevels=maxDepth, initializationType='atomic',divideCriterion=divideCriterion, divideParameter=divideParameter, printTreeProperties=True,onlyFillOne=False)
    
    start = time.time()
    tree.orthonormalizeOrbitals(targetOrbital=3, external=False)
    internalTime = time.time()-start
    
    print('\n\nTime for internal orthogonalization: ', internalTime)
    
    sources = tree.extractPhi(0)
    phiA0 = sources[:,3]
    sources = tree.extractPhi(3)
    phiA3 = sources[:,3]
    
    start = time.time()
    tree.orthonormalizeOrbitals(targetOrbital=3, external=True)
    externalTime = time.time()-start

    print('Time for external orthogonalization: ', externalTime)


    sources = tree.extractPhi(0)
    phiB0 = sources[:,3]
    sources = tree.extractPhi(3)
    phiB3 = sources[:,3]
    
    print('Max diff between internal and external: ', np.max( np.abs(phiA0 - phiB0 )))
    print('Max diff between internal and external: ', np.max( np.abs(phiA3 - phiB3 )))
    

def meshDistributions(domain,order,minDepth, maxDepth, additionalDepthAtAtoms, divideCriterion, 
                          divideParameter1, divideParameter2=0.0, divideParameter3=0.0, divideParameter4=0.0, 
                          smoothingEpsilon=0.0, base=1.0, causeFigure=False,
                          inputFile='',savedMesh=''):    
    
    
    divideParameter1 *= base
    divideParameter2 *= base
    divideParameter3 *= base
    divideParameter4 *= base
    
    [coordinateFile, referenceEigenvaluesFile, DummyOutputFile] = np.genfromtxt(inputFile,dtype="|U100")[:3]
    [Eband, Ekinetic, Eexchange, Ecorrelation, Eelectrostatic, Etotal] = np.genfromtxt(inputFile)[3:]


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
    print('nElectrons = ', nElectrons)
    print('nOrbitals  = ', nOrbitals)
    print([coordinateFile, Etotal, Eexchange, Ecorrelation, Eband])
    tree = Tree(-domain,domain,order,-domain,domain,order,-domain,domain,order,nElectrons,nOrbitals,additionalDepthAtAtoms=additionalDepthAtAtoms,minDepth=minDepth,gaugeShift=gaugeShift,
                coordinateFile=coordinateFile,smoothingEps=smoothingEpsilon,inputFile=inputFile)#, iterationOutFile=outputFile)

    
    print('max depth ', maxDepth)
    tree.buildTree( maxLevels=maxDepth, initializationType='atomic',divideCriterion=divideCriterion, 
                    divideParameter1=divideParameter1, divideParameter2=divideParameter2, divideParameter3=divideParameter3, divideParameter4=divideParameter4, 
                    savedMesh=savedMesh,printTreeProperties=True,onlyFillOne=False)
    
    print(tree.levelCounts)
    print()
    print(tree.levelCounts.keys())
    print()
    print(tree.levelCounts.values())
    maxHeight = np.max(list(tree.levelCounts.values()))
    plt.figure()
    plt.bar(list(tree.levelCounts.keys()),list(tree.levelCounts.values()) )
    plt.xlabel('Refinement Depth')
    plt.ylabel('Number of Cells')
    if divideCriterion=='Krasny':
        plt.title('Mesh Type: 4 parameters (%1.2f,%1.2f,%1.2f,%1.2f)' %(divideParameter1,divideParameter2,divideParameter3,divideParameter4))
    elif divideCriterion=='Nathan':
        plt.title('Mesh Type: 2 parameters (%1.2f,%1.2f)' %(divideParameter1,divideParameter2))
    elif divideCriterion=='LW5':
        plt.title('Mesh Type: LW5 - %1.2f' %(divideParameter1))
    elif divideCriterion=='Krasny_density':
        plt.title('Mesh Type: HOMO LW5 - %1.2f' %(divideParameter1))
        
    if causeFigure==True:
        fig, axes = plt.subplots(2, 2)
        axes[0,0].bar(list(tree.criteria1.keys()),list(tree.criteria1.values()) )
        axes[0,1].bar(list(tree.criteria2.keys()),list(tree.criteria2.values()) )
        axes[1,0].bar(list(tree.criteria3.keys()),list(tree.criteria3.values()) )
        axes[1,1].bar(list(tree.criteria4.keys()),list(tree.criteria4.values()) )
            
        axes[0,0].set_title('Criteria 1')
        axes[0,1].set_title('Criteria 2')
        axes[1,0].set_title('Criteria 3')
        axes[1,1].set_title('Criteria 4')
            
        axes[0,0].set_ylim([0, int(1.1*maxHeight)])
        axes[0,1].set_ylim([0, int(1.1*maxHeight)])
        axes[1,0].set_ylim([0, int(1.1*maxHeight)])
        axes[1,1].set_ylim([0, int(1.1*maxHeight)])
        plt.tight_layout()
  
    plt.savefig('distribution.png', bbox_inches='tight')
    plt.show()

def densityInterpolation(xi,yi,zi,xf,yf,zf,numpts,
                         domain,order,minDepth, maxDepth, additionalDepthAtAtoms, divideCriterion, 
                          divideParameter1, divideParameter2=0.0, divideParameter3=0.0, divideParameter4=0.0, 
                          smoothingEpsilon=0.0, base=1.0,
                          inputFile=''):    
    
    
    divideParameter1 *= base
    divideParameter2 *= base
    divideParameter3 *= base
    divideParameter4 *= base
    
    [coordinateFile, referenceEigenvaluesFile, DummyOutputFile] = np.genfromtxt(inputFile,dtype="|U100")[:3]
    [Eband, Ekinetic, Eexchange, Ecorrelation, Eelectrostatic, Etotal] = np.genfromtxt(inputFile)[3:]


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
    print('nElectrons = ', nElectrons)
    print('nOrbitals  = ', nOrbitals)
    print([coordinateFile, Etotal, Eexchange, Ecorrelation, Eband])
    tree = Tree(-domain,domain,order,-domain,domain,order,-domain,domain,order,nElectrons,nOrbitals,additionalDepthAtAtoms=additionalDepthAtAtoms,minDepth=minDepth,gaugeShift=gaugeShift,
                coordinateFile=coordinateFile,smoothingEps=smoothingEpsilon,inputFile=inputFile)#, iterationOutFile=outputFile)

    
    print('max depth ', maxDepth)
    tree.buildTree( maxLevels=maxDepth, initializationType='atomic',divideCriterion=divideCriterion, 
                    divideParameter1=divideParameter1, divideParameter2=divideParameter2, divideParameter3=divideParameter3, divideParameter4=divideParameter4, 
                    printTreeProperties=True,onlyFillOne=False,restart=True)
    
    r, rho = tree.interpolateDensity(xi,yi,zi,xf,yf,zf,numpts,plot=False)
    
    
    initialRho = np.zeros_like(r)
    x = np.linspace(xi,xf,numpts)
    y = np.linspace(yi,yf,numpts)
    z = np.linspace(zi,zf,numpts)
    for atom in tree.atoms:
        rtemp = np.sqrt( (x-atom.x)**2 + (y-atom.y)**2 + (z-atom.z)**2 )
        try:
            initialRho += atom.interpolators['density'](rtemp)
        except ValueError:
            initialRho += 0.0   # if outside the interpolation range, assume 0.
            
    plt.figure()
    plt.semilogy(r,rho,'bo')
#     plt.semilogy(r,initialRho,'rx')
    plt.title('Density along Line')
    
    plt.figure()
    plt.semilogy(r,abs( rho-initialRho )/initialRho,'ko')
    plt.title('Relative Error in Interpolation from Cells and Interpolation from Initial Data')
    plt.show()
    
    
def plot_LW_density():
    r = np.linspace(1e-1,1,1000)
    
#     plt.rc('text', usetex=True)
#     plt.rc('font', family='serif')
    
    plt.figure()
    for N in [5]:
        density = np.zeros(len(r))
        if N==3:
            A = 648.82
            c = [52, -102, 363, 1416, 4164, 5184, 2592]
        elif N == 4:
            A = 1797.9
            c = [423,-1286,2875,16506,79293,292512,611136,697320, 348660]
        elif N==5:
            A = 3697.1
            c = [2224, -9018, 16789, 117740, 733430, 3917040, 16879920, 49186500, 91604250, 100516500, 50258250]
        for k in range(0,2*N+1):
            density += c[k] * r**(-k)
        
        density *= np.exp(-2*r)
        density = 1000/A*density**(3/(2*N+3))
    
        
        plt.plot(r,density, label = 'N = %i'%N)
    
    nathanDensity = np.zeros_like(r)
    for i in range(len(nathanDensity)):
        nathanDensity[i] = meshDensity(r[i],4,'Nathan_density')
    plt.plot(r,nathanDensity,label='Nathan')
        
#         plt.plot(r, 1000/3697.1*(np.exp(-2*r)*50258250/r**10)**(3/13), label='exp(-2*r)/r_sq')
#     k = np.sqrt(2*0.2)
#     plt.plot(r, 1000/3697.1*exp(-k*r)* (2224 - 9018/r + 16789/r**2 + 117740/r**3 + 733430/r**4 + 3917040/r**5 + 16879920/r**6
#                + 49186500/r**7 + 91604250/r**8 + 100516500/r**9 + 50258250/r**10) **(3/13), label='2')
#         plt.plot(r, 1000/3697.1*(np.exp(-2*r)* (50258250/r**10) )**(3/13), label='LW5_truncated')
        

#     plt.plot(r,(density[0] / (1/r[0]) ) *1/r, label='1/r')
#     plt.plot(r,(density[0] / (np.exp(-r[0])/r[0]) ) *np.exp(-r)/r, label='exp(-r)/r')
#     k = np.sqrt(2*0.2)
#     plt.plot(r,(density[0] / (np.exp(-k*r[0])/r[0]**2) ) *np.exp(-k*r)/r**2, label='exp(-k*r)/r_sq')
#     plt.plot(r,(density[0] / (np.exp(-2*r[0])/r[0]**2) ) *np.exp(-2*r)/r**2, label='exp(-2*r)/r_sq')
#     plt.plot(r,(density[0] / (r[0] + np.exp(-k*r[0])/r[0]**2) ) *(r+np.exp(-k*r))/r**2, label='(r+exp(-k*r))/r_sq')
    plt.legend()
    plt.xlabel('Distance from nucleus')
    plt.ylabel('LW Mesh Density')
    plt.title('Mesh Density Functions')
    plt.show()

if __name__ == "__main__":
    gaugeShift=-0.5
#     plot_LW_density()
#     densityInterpolation(-6.1,1,0,6.1,1,0,1000,
#                     domain=20,order=5,
#                     minDepth=3, maxDepth=20, additionalDepthAtAtoms=0, divideCriterion='LW5', 
#                     divideParameter1=1500, divideParameter2=10.1/1, divideParameter3=100, divideParameter4=100,
#                     smoothingEpsilon=0.0,base=1.0, inputFile='../src/utilities/molecularConfigurations/carbonMonoxideAuxiliary.csv')
    
    
#     meshDistributions(domain=20,order=5,
#                         minDepth=3, maxDepth=20, additionalDepthAtAtoms=0, divideCriterion='ParentChildrenIntegral', 
#                         divideParameter1=1e6, divideParameter2=1e6, divideParameter3=1e-6, divideParameter4=0,
#                         smoothingEpsilon=0.0,base=1.0, causeFigure=False, inputFile='../src/utilities/molecularConfigurations/oxygenAtomAuxiliary.csv',
#                         savedMesh='benzene_1e-6_rotated.npy')
    
    
#     timingTestsForOrbitalInitializations(domain=20,order=5,
#                           minDepth=3, maxDepth=20, divideCriterion='LW5', 
#                           divideParameter=1500,inputFile='../src/utilities/molecularConfigurations/oxygenAtomAuxiliary.csv')


#     timingTestsForOrbitalOrthogonalizations(domain=20,order=4,
#                           minDepth=3, maxDepth=20, divideCriterion='LW5', 
#                           divideParameter=500,inputFile='../src/utilities/molecularConfigurations/oxygenAtomAuxiliary.csv')
    
    # param1: wavefunction variation
    # param2: wavefunction integral
    # param3: density integral   
    # param4: Vext integral  
    
    
    # ParentChildrenIntegral
#     tree, tree2 = testTreeSaveAndReconstruction(domain=20,order=5,
#                         minDepth=3, maxDepth=20, additionalDepthAtAtoms=1, divideCriterion='ParentChildrenIntegral', 
#                         divideParameter1=1e6, divideParameter2=1e6, divideParameter3=1e-5, divideParameter4=0,
#                         smoothingEpsilon=0.0,inputFile='../src/utilities/molecularConfigurations/berylliumAuxiliary.csv', 
#                         outputFile='/Users/nathanvaughn/Desktop/meshTests/benzene/PCI') 
    
#     for i in range(len(tree2.saveList)):
#         if tree.saveList[i] != tree2.saveList[i]:
#             print('i = ', i)
#             print('tree', tree.saveList[i])
#             print('tree2', tree2.saveList[i])
            
            
            
#             oxygenAtomAuxiliary
     
    
#     testDask(domain=20,order=5,
#             minDepth=3, maxDepth=20, additionalDepthAtAtoms=0, divideCriterion='LW5', 
#             divideParameter1=500, divideParameter2=1e6, divideParameter3=3e-5, divideParameter4=0,
#             smoothingEpsilon=0.0,inputFile='../src/utilities/molecularConfigurations/oxygenAtomAuxiliary.csv', 
#             outputFile='/Users/nathanvaughn/Desktop/meshTests/O2/aspectRatioTesting',
#             savedMesh='') 
    
    #ParentChildrenIntegral
#     tree = exportMeshForParaview(domain=30,order=5,
#                         minDepth=3, maxDepth=20, additionalDepthAtAtoms=0, divideCriterion='ParentChildrenIntegral', 
#                         divideParameter1=1500, divideParameter2=0, divideParameter3=1e-3, divideParameter4=3,
#                         smoothingEpsilon=0.0,inputFile='../src/utilities/molecularConfigurations/benzeneAuxiliary.csv', 
#                         outputFile='/Users/nathanvaughn/Desktop/meshTests/forPaper/benzene_1e-3_rotated2D',
#                         savedMesh='')   
#     
#     tree = exportMeshForParaview(domain=30,order=5,
#                         minDepth=3, maxDepth=20, additionalDepthAtAtoms=0, divideCriterion='ParentChildrenIntegral', 
#                         divideParameter1=1500, divideParameter2=0, divideParameter3=1e-4, divideParameter4=4,
#                         smoothingEpsilon=0.0,inputFile='../src/utilities/molecularConfigurations/benzeneAuxiliary.csv', 
#                         outputFile='/Users/nathanvaughn/Desktop/meshTests/forPaper/benzene_1e-4_rotated2D',
#                         savedMesh='')   
#     
#     tree = exportMeshForParaview(domain=30,order=5,
#                         minDepth=3, maxDepth=20, additionalDepthAtAtoms=0, divideCriterion='ParentChildrenIntegral', 
#                         divideParameter1=1500, divideParameter2=0, divideParameter3=1e-5, divideParameter4=5,
#                         smoothingEpsilon=0.0,inputFile='../src/utilities/molecularConfigurations/benzeneAuxiliary.csv', 
#                         outputFile='/Users/nathanvaughn/Desktop/meshTests/forPaper/benzene_1e-5_rotated2D',
#                         savedMesh='')  
    
    
#     ## THIS WAS USED TO GENERATE FIGURES IN PAPER
#     tree = exportMeshForParaview(domain=30,order=5,
#                         minDepth=3, maxDepth=20, additionalDepthAtAtoms=0, divideCriterion='ParentChildrenIntegral', 
#                         divideParameter1=1500, divideParameter2=0, divideParameter3=1e-7, divideParameter4=4,
#                         smoothingEpsilon=0.0,inputFile='../src/utilities/molecularConfigurations/benzeneAuxiliary.csv', 
#                         outputFile='/Users/nathanvaughn/Desktop/meshTests/forPaper/benzene_1e-7_rotated2D',
#                         savedMesh='')   

#     tree = exportMeshForParaview(domain=20,order=5,
#                         minDepth=3, maxDepth=20, additionalDepthAtAtoms=0, divideCriterion='ParentChildrenIntegral', 
#                         divideParameter1=1500, divideParameter2=0, divideParameter3=1e-4, divideParameter4=4,
#                         smoothingEpsilon=0.0,inputFile='../src/utilities/molecularConfigurations/berylliumAuxiliary.csv', 
#                         outputFile='/Users/nathanvaughn/Desktop/meshTests/forVisitTesting/beryllium',
#                         savedMesh='') 
    
    tree = exportMeshForParaview(domain=20,order=5,
                        minDepth=3, maxDepth=20, additionalDepthAtAtoms=0, divideCriterion='LW5', 
                        divideParameter1=1500, divideParameter2=0, divideParameter3=1e-2, divideParameter4=4,
                        smoothingEpsilon=0.0,inputFile='../src/utilities/molecularConfigurations/berylliumAuxiliary.csv', 
                        outputFile='/Users/nathanvaughn/Desktop/meshTests/forVisitTesting/beryllium',
                        savedMesh='') 
    
     
#                         savedMesh='benzene_1e-6_rotated.npy')        
#                         savedMesh='ParentChildrenIntegral_500_0_0.0001_0.npy')        
             
#     tree = exportMeshForParaview(domain=30,order=5,
#                         minDepth=3, maxDepth=20, additionalDepthAtAtoms=0, divideCriterion='ParentChildrenIntegral', 
#                         divideParameter1=500, divideParameter2=999, divideParameter3=1e-6, divideParameter4=0,
#                         smoothingEpsilon=0.0,inputFile='../src/utilities/molecularConfigurations/benzeneAuxiliary.csv', 
#                         outputFile='/Users/nathanvaughn/Desktop/meshTests/CO/PCI_Benzene_reconstructed',
#                         savedMesh='')
    
#     tree = exportMeshToCompareDensity(domain=30,order=5,
#                         minDepth=3, maxDepth=20, additionalDepthAtAtoms=0, divideCriterion='ParentChildrenIntegral', 
#                         divideParameter1=500, divideParameter2=999, divideParameter3=1e-6, divideParameter4=0,
#                         smoothingEpsilon=0.0,inputFile='../src/utilities/molecularConfigurations/benzeneAuxiliary.csv', 
#                         outputFile='/Users/nathanvaughn/Desktop/meshTests/CO/PCI_Benzene_reconstructed',
#                         savedMesh='ParentChildrenIntegral_500_999_1e-06_0__1416000.npy', 
#                         restartFilesDir='/Users/nathanvaughn/Documents/synchronizedDataFiles/restartFiles_1416000_after25/',
#                         iterationNumber=10)
#     
#     exportMeshForParaview(domain=31,order=3,
#                         minDepth=3, maxDepth=20, additionalDepthAtAtoms=0, divideCriterion='LW5', 
#                         divideParameter1=500, divideParameter2=1e6, divideParameter3=1e-5, divideParameter4=0,
#                         smoothingEpsilon=0.0,inputFile='../src/utilities/molecularConfigurations/benzeneAuxiliary.csv', 
#                         outputFile='/Users/nathanvaughn/Desktop/meshTests/benzene/LW5')
#     


#     exportMeshForTreecodeTesting(domain=30,order=5,
#                         minDepth=3, maxDepth=20, additionalDepthAtAtoms=0, divideCriterion='ParentChildrenIntegral', 
#                         divideParameter1=0, divideParameter2=0, divideParameter3=1e-5, divideParameter4=0,
#                         inputFile='../src/utilities/molecularConfigurations/benzeneAuxiliary.csv')
#     
#     exportMeshForTreecodeTesting(domain=30,order=5,
#                         minDepth=3, maxDepth=20, additionalDepthAtAtoms=0, divideCriterion='ParentChildrenIntegral', 
#                         divideParameter1=0, divideParameter2=0, divideParameter3=1e-6, divideParameter4=0,
#                         inputFile='../src/utilities/molecularConfigurations/benzeneAuxiliary.csv')
#     
#     exportMeshForTreecodeTesting(domain=30,order=5,
#                         minDepth=3, maxDepth=20, additionalDepthAtAtoms=0, divideCriterion='ParentChildrenIntegral', 
#                         divideParameter1=0, divideParameter2=0, divideParameter3=1e-7, divideParameter4=0,
#                         inputFile='../src/utilities/molecularConfigurations/benzeneAuxiliary.csv')
    
#for dp3 in [1e-4, 1e-5, 1e-6, 1e-7, 1e-8]:
# for dp3 in [1e-2, 1e-3, 3e-8]:
#     exportMeshForTreecodeTesting(   domain=30,order=5,
#                                     minDepth=3, maxDepth=20, additionalDepthAtAtoms=0, divideCriterion='ParentChildrenIntegral', 
#                                     divideParameter1=0, divideParameter2=0, divideParameter3=dp3, divideParameter4=0,
#                                     inputFile='../src/utilities/molecularConfigurations/benzeneAuxiliary.csv')
    
#                         divideParameter=1e-5,inputFile='../src/utilities/molecularConfigurations/hydrogenMoleculeAuxiliary.csv')
#                         divideParameter1=1.0, divideParameter2=1.0,inputFile='../src/utilities/molecularConfigurations/oxygenAtomAuxiliary.csv')
#                         divideParameter1=500,inputFile='../src/utilities/molecularConfigurations/oxygenAtomAuxiliary.csv')
#                         divideParameter=1e-3,inputFile='../src/utilities/molecularConfigurations/berylliumAuxiliary.csv')
#                         divideParameter=1.25e-3,inputFile='../src/utilities/molecularConfigurations/carbonMonoxideAuxiliary.csv')
     
    
