import sys
import mpi4py.MPI as MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
import numpy as np

from mpiUtilities import rprint, global_dot
sys.path.append('../dataStructures')
from TreeStruct_CC import Tree




def inializeBaseMesh(XL,YL,ZL,maxSideLength,verbose=0):
    '''
    Input the domain parameters and a maximum cell size.  Return a list of the minimally refined mesh coordinates.
    
    :param XL:
    :param YL:
    :param ZL:
    :param maxSideLength:
    '''
    
    nx = -( (-2*XL) // maxSideLength )  # give the number of base cells in the x direction
    ny = -( (-2*YL) // maxSideLength )  # give the number of base cells in the y direction
    nz = -( (-2*ZL) // maxSideLength )  # give the number of base cells in the z direction
    
    if verbose>0: print('nx, ny, nz: ', nx,ny,nz)
    x = np.linspace(-XL,XL,nx+1)
    y = np.linspace(-YL,YL,ny+1)
    z = np.linspace(-ZL,ZL,nz+1)
    
    cells = []
    
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                cells.append( [x[i], x[i+1], y[j], y[j+1], z[k], z[k+1]] )
    
#     rprint("Number of coarse cells: ", len(cells))
#     for i in range(len(cells)):
#         print(cells[i])
    return cells

def buildMeshFromMinimumDepthCells(XL,YL,ZL,maxSideLength,inputFile,outputFile,srcdir,order,gaugeShift,divideCriterion='ParentChildrenIntegral',divideParameter=1):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    
    cells=inializeBaseMesh(XL,YL,ZL,maxSideLength)
    
    x=np.empty(0)
    y=np.empty(0)
    z=np.empty(0)
    w=np.empty(0)
    for i in range(len(cells)):
        if i%size==rank:
            
            X,Y,Z,W,atoms,nPoints,nOrbitals,nElectrons,referenceEigenvalues = refineCell(cells[i],inputFile,outputFile,srcdir,order,gaugeShift,divideCriterion=divideCriterion,
                                                                                         divideParameter1=divideParameter, divideParameter2=divideParameter, divideParameter3=divideParameter, divideParameter4=divideParameter)
            x=np.append(x,X)
            y=np.append(y,Y)
            z=np.append(z,Z)
            w=np.append(w,W)
    return x,y,z,w,atoms,nPoints,nOrbitals,nElectrons,referenceEigenvalues

def refineCell(coordinates,inputFile,outputFile,srcdir,order,gaugeShift,additionalDepthAtAtoms=0,minDepth=0,divideCriterion='ParentChildrenIntegral',divideParameter1=0,divideParameter2=0,divideParameter3=0,divideParameter4=0, verbose=0):
    '''
    setUp() gets called before every test below.
    '''
    [xmin, xmax, ymin, ymax, zmin, zmax] = coordinates
    if verbose>0: print(xmin,xmax,ymin,ymax,zmin,zmax)

    [coordinateFile, referenceEigenvaluesFile, DummyOutputFile] = np.genfromtxt(inputFile,dtype="|U100")[:3]
    [Eband, Ekinetic, Eexchange, Ecorrelation, Eelectrostatic, Etotal] = np.genfromtxt(inputFile)[3:]
    
    if verbose>0: rprint('Reading atomic coordinates from: ', coordinateFile)
    atomData = np.genfromtxt(srcdir+coordinateFile,delimiter=',',dtype=float)
#     rprint(atomData)
    if np.shape(atomData)==(5,):
        nElectrons = atomData[3]
    else:
        nElectrons = 0
        for i in range(len(atomData)):
            nElectrons += atomData[i,3] 
    
    
#     nOrbitals = int( np.ceil(nElectrons/2)  ) + 2
    nOrbitals = int( np.ceil(nElectrons/2)  )   # start with the minimum number of orbitals 
#     nOrbitals = int( np.ceil(nElectrons/2) + 1 )   # start with the minimum number of orbitals plus 1.   
                                            # If the final orbital is unoccupied, this amount is enough. 
                                            # If there is a degeneracy leading to teh final orbital being 
                                            # partially filled, then it will be necessary to increase nOrbitals by 1.
                        
    # For O2, init 10 orbitals.
#     nOrbitals=10                    

    occupations = 2*np.ones(nOrbitals)


    if inputFile==srcdir+'utilities/molecularConfigurations/oxygenAtomAuxiliary.csv':
        nOrbitals=5
        occupations = 2*np.ones(nOrbitals)
        occupations[2] = 4/3
        occupations[3] = 4/3
        occupations[4] = 4/3
        rprint('For oxygen atom, nOrbitals = ', nOrbitals)
        
    elif inputFile==srcdir+'utilities/molecularConfigurations/benzeneAuxiliary.csv':
        nOrbitals=27
        occupations = 2*np.ones(nOrbitals)
        for i in range(21,nOrbitals):
            occupations[i]=0 
        
        
    elif inputFile==srcdir+'utilities/molecularConfigurations/O2Auxiliary.csv':
        nOrbitals=10
        occupations = [2,2,2,2,4/3,4/3,4/3,4/3,4/3,4/3]
        
    elif inputFile==srcdir+'utilities/molecularConfigurations/carbonMonoxideAuxiliary.csv':
#         nOrbitals=10
#         occupations = [2, 2, 4/3 ,4/3 ,4/3, 
#                        2, 2, 2/3 ,2/3 ,2/3 ]
        nOrbitals=7
        occupations = 2*np.ones(nOrbitals)
    
    elif inputFile==srcdir+'utilities/molecularConfigurations/hydrogenMoleculeAuxiliary.csv':
        nOrbitals=1
        occupations = [2]
    
#     print('inputFile == '+inputFile)
#     return
    if verbose>0: rprint([coordinateFile, outputFile, nElectrons, nOrbitals, 
                          Etotal, Eexchange, Ecorrelation, Eband, gaugeShift])
    smoothingEps=0.0
    maxDepth=20
    savedMesh=''
    restart=False
    referenceEigenvalues = np.array( np.genfromtxt(srcdir+referenceEigenvaluesFile,delimiter=',',dtype=float) )
    if verbose>0: rprint(referenceEigenvalues)
    if verbose>0: rprint(np.shape(referenceEigenvalues))
    tree = Tree(xmin,xmax,order,ymin,ymax,order,zmin,zmax,order,nElectrons,nOrbitals,additionalDepthAtAtoms=additionalDepthAtAtoms,minDepth=minDepth,gaugeShift=gaugeShift,
                coordinateFile=srcdir+coordinateFile,smoothingEps=smoothingEps, inputFile=srcdir+inputFile)#, iterationOutFile=outputFile)
    tree.referenceEigenvalues = np.copy(referenceEigenvalues)
    tree.occupations = occupations
   
    
    if verbose>0: rprint('max depth ', maxDepth)
#     print(tree.atoms)
#     print(tree.root.gridpoints)
#     tree.buildTree_FirstAndSecondKind( maxLevels=maxDepth, initializationType='atomic',divideCriterion=divideCriterion, 
    tree.buildTree( maxLevels=maxDepth, initializationType='atomic',divideCriterion=divideCriterion, 
                    divideParameter1=divideParameter1, divideParameter2=divideParameter2, divideParameter3=divideParameter3, divideParameter4=divideParameter4, 
                    savedMesh=savedMesh, restart=restart, printTreeProperties=False,onlyFillOne=False)

 
#     return 
#     X,Y,Z,W,RHO,orbitals = tree.extractXYZ()
    X,Y,Z,W,RHO, XV, YV, ZV, vertexIdx, centerIdx, ghostCells = tree.extractXYZ()

#     
    atoms = tree.atoms
    nPoints = len(X)

    tree=None
#     rprint("Returning from setupTree.")
    return X,Y,Z,W,atoms,nPoints,nOrbitals,nElectrons,referenceEigenvalues


if __name__=="__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    cells = inializeBaseMesh(20,20,20,7)
    
#     n=15
#     
#     
#     
#     if rank==0:
#         x = np.random.random(n)
#         y = np.random.random(n)
#         z = np.random.random(n)
#         w = np.random.random(n)
#     else:
#         x = np.empty(0)
#         y = np.empty(0)
#         z = np.empty(0)
#         w = np.empty(0)
#     
#     scatterArrays(x,y,z,w,comm)
    