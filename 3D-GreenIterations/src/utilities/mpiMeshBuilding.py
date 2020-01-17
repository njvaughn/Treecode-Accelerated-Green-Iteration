import sys
import mpi4py.MPI as MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
import numpy as np

from mpiUtilities import rprint, global_dot
sys.path.append('../dataStructures')
from TreeStruct_CC import Tree
from AtomStruct import Atom




def inializeBaseMesh(XL,YL,ZL,maxSideLength,verbose=1):
    '''
    Input the domain parameters and a maximum cell size.  Return a list of the minimally refined mesh coordinates.
    
    :param XL:
    :param YL:
    :param ZL:
    :param maxSideLength:
    '''
    
    nx = int( -( (-2*XL) // maxSideLength ) )  # give the number of base cells in the x direction
    ny = int( -( (-2*YL) // maxSideLength ) )  # give the number of base cells in the y direction
    nz = int( -( (-2*ZL) // maxSideLength ) )  # give the number of base cells in the z direction
    

    if verbose>0: rprint(rank,'nx, ny, nz: %i, %i, %i' %(nx,ny,nz))
    x = np.linspace(-XL,XL,nx+1)
    y = np.linspace(-YL,YL,ny+1)
    z = np.linspace(-ZL,ZL,nz+1)
    
    cells = []
    
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                cells.append( [x[i], x[i+1], y[j], y[j+1], z[k], z[k+1]] )
    
    if verbose>0: rprint(rank,"Number of coarse cells: ", len(cells))
#     for i in range(len(cells)):
#         print(cells[i])
    return cells

def buildMeshFromMinimumDepthCells(XL,YL,ZL,maxSideLength,coreRepresentation,inputFile,outputFile,srcdir,order,gaugeShift,
                                   divideCriterion,divideParameter1,divideParameter2,divideParameter3,divideParameter4,verbose=0):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    ## Setup atoms and PSP structs if needed.
    [coordinateFile, referenceEigenvaluesFile, DummyOutputFile] = np.genfromtxt(inputFile,dtype="|U100")[:3]
    if verbose>-1: rprint(rank,'Reading atomic coordinates from: ', coordinateFile)
    atomData = np.genfromtxt(srcdir+coordinateFile,delimiter=',',dtype=float)
    nElectrons=0
    nOrbitals=0
    PSPs = {}  # dictionary of PSPs for each atomic species, should this be a Pseudopotential calculation
    if np.shape(atomData)==(5,):
        atoms = np.empty((1,),dtype=object)
        atom = Atom(atomData[0],atomData[1],atomData[2],atomData[3],atomData[4],coreRepresentation)
        atoms[0] = atom
        nOrbitals += int(atomData[4])
        if coreRepresentation=="AllElectron":
            nElectrons+=atomData[3]
        elif coreRepresentation=="Pseudopotential":
            atom.setPseudopotentialObject(PSPs)
            nElectrons += atom.PSP.psp['header']['z_valence']
    else:
        atoms = np.empty((len(atomData),),dtype=object)
        for i in range(len(atomData)):
            atom = Atom(atomData[i,0],atomData[i,1],atomData[i,2],atomData[i,3],atomData[i,4],coreRepresentation)
            atoms[i] = atom
            nOrbitals += int(atomData[i,4])
            if coreRepresentation=="AllElectron":
                nElectrons+=atomData[i,3]
            elif coreRepresentation=="Pseudopotential":
                atom.setPseudopotentialObject(PSPs)
                nElectrons += atom.PSP.psp['header']['z_valence']
            else:
                print("What is coreRepresentation?")
                exit(-1)
    
    

    
    
#     nOrbitals = int( np.ceil(nElectrons/2)*1.2  )   # start with the minimum number of orbitals 
    occupations = 2*np.ones(nOrbitals)

    if verbose>0: rprint(rank,[coordinateFile, outputFile, nElectrons, nOrbitals, 
                          Etotal, Eexchange, Ecorrelation, Eband, gaugeShift])
    
    
    cells=inializeBaseMesh(XL,YL,ZL,maxSideLength)
    
    x=np.empty(0)
    y=np.empty(0)
    z=np.empty(0)
    w=np.empty(0)
    for i in range(len(cells)):
        if i%size==rank:
#             print("CALLING refineCell ==================================================")
            X,Y,Z,W,atoms,nPoints,nOrbitals,nElectrons,referenceEigenvalues = refineCell(nElectrons,nOrbitals,atoms,coreRepresentation,cells[i],inputFile,outputFile,srcdir,order,gaugeShift,divideCriterion=divideCriterion,
                                                                                         divideParameter1=divideParameter1, divideParameter2=divideParameter2, divideParameter3=divideParameter3, divideParameter4=divideParameter4)
            x=np.append(x,X)
            y=np.append(y,Y)
            z=np.append(z,Z)
            w=np.append(w,W)
    return x,y,z,w,atoms,PSPs,nPoints,nOrbitals,nElectrons,referenceEigenvalues

def refineCell(nElectrons,nOrbitals,atoms,coreRepresentation,coordinates,inputFile,outputFile,srcdir,order,gaugeShift,additionalDepthAtAtoms=0,minDepth=0,divideCriterion='ParentChildrenIntegral',divideParameter1=0,divideParameter2=0,divideParameter3=0,divideParameter4=0, verbose=0):
    '''
    setUp() gets called before every test below.
    '''
    [xmin, xmax, ymin, ymax, zmin, zmax] = coordinates
    if verbose>0: print(xmin,xmax,ymin,ymax,zmin,zmax)

    [coordinateFile, referenceEigenvaluesFile, DummyOutputFile] = np.genfromtxt(inputFile,dtype="|U100")[:3]
    [Eband, Ekinetic, Eexchange, Ecorrelation, Eelectrostatic, Etotal] = np.genfromtxt(inputFile)[3:]
    
    
    savedMesh=''
    restart=False
    referenceEigenvalues = np.array( np.genfromtxt(srcdir+referenceEigenvaluesFile,delimiter=',',dtype=float) )
    if verbose>0: rprint(rank,referenceEigenvalues)
    if verbose>0: rprint(rank,np.shape(referenceEigenvalues))
    tree = Tree(xmin,xmax,order,ymin,ymax,order,zmin,zmax,order,atoms,coreRepresentation,nElectrons,nOrbitals,additionalDepthAtAtoms=additionalDepthAtAtoms,minDepth=minDepth,gaugeShift=gaugeShift,
                coordinateFile=srcdir+coordinateFile, inputFile=srcdir+inputFile)#, iterationOutFile=outputFile)

   
    tree.buildTree( initializationType='atomic',divideCriterion=divideCriterion, 
                    divideParameter1=divideParameter1, divideParameter2=divideParameter2, divideParameter3=divideParameter3, divideParameter4=divideParameter4, 
                    savedMesh=savedMesh, restart=restart, printTreeProperties=False,onlyFillOne=False)
    
    tree.finalDivideBasedOnNuclei(coordinateFile)
    
    tree.exportGridpoints
    X,Y,Z,W,RHO, XV, YV, ZV, vertexIdx, centerIdx, ghostCells = tree.extractXYZ()

#     
    atoms = tree.atoms
    nPoints = len(X)

    tree=None
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
    