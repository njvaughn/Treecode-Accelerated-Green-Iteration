import sys
import mpi4py.MPI as MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
import numpy as np

from mpiUtilities import rprint, global_dot
from loadBalancer import loadBalance
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
    volumes=[]
    
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                cells.append( [x[i], x[i+1], y[j], y[j+1], z[k], z[k+1]] )
                volumes.append( (x[i+1]-x[i])*(y[j+1]-y[j])*(z[k+1]-z[k]) )
    
    if verbose>0: rprint(rank,"Number of coarse cells: ", len(cells))
    if verbose>0: rprint(rank, "Expected volume:   ", (2*XL*2*YL*2*ZL))
    if verbose>0: rprint(rank, "Cumulative volume: ", np.sum(volumes))
    assert abs((2*XL*2*YL*2*ZL) - np.sum(volumes)) < 1e-12, "base mesh cells volumes do not add up to the expected total volume."
    
#     for i in range(len(cells)):
#         print(cells[i])
#     exit(-1)
    return cells

def inializeBaseMesh_distributed(XL,YL,ZL,maxSideLength,verbose=1):
    '''
    Input the domain parameters and a maximum cell size.  Return a list of the minimally refined mesh coordinates.
    
    :param XL:
    :param YL:
    :param ZL:
    :param maxSideLength:
    '''
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    nx = int( -( (-2*XL) // maxSideLength ) )  # give the number of base cells in the x direction
    ny = int( -( (-2*YL) // maxSideLength ) )  # give the number of base cells in the y direction
    nz = int( -( (-2*ZL) // maxSideLength ) )  # give the number of base cells in the z direction
    

    if verbose>0: rprint(rank,'nx, ny, nz: %i, %i, %i' %(nx,ny,nz))
    x = np.linspace(-XL,XL,nx+1)
    y = np.linspace(-YL,YL,ny+1)
    z = np.linspace(-ZL,ZL,nz+1)
    
    cells = []
    cellsX = []
    cellsY = []
    cellsZ = []
    volumes=[]
    idx=0
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                
                if idx%size==rank:
                    cells.append( [x[i], x[i+1], y[j], y[j+1], z[k], z[k+1]] )
                    volumes.append( (x[i+1]-x[i])*(y[j+1]-y[j])*(z[k+1]-z[k]) )
                    
                    cellsX.append( (x[i+1]+x[i])/2 )
                    cellsY.append( (y[j+1]+y[j])/2 )
                    cellsZ.append( (z[k+1]+z[k])/2 )
                
                idx+=1
    
    if verbose>0: print("Rank %i, number of coarse cells: %i, volume = %f" %(rank, len(cells), np.sum(volumes)) )
#     if verbose>0: rprint(rank,"Rank %i, number of coarse cells: %i" %(rank, len(cells)) )
#     if verbose>0: rprint(rank, "Expected volume:   ", (2*XL*2*YL*2*ZL))
#     if verbose>0: rprint(rank, "Cumulative volume: ", np.sum(volumes))

    totalVolume = comm.allreduce(np.sum(volumes))
    rprint(rank,"Total volume: ", totalVolume)
    assert abs((2*XL*2*YL*2*ZL) - totalVolume) < 1e-12, "base mesh cells volumes do not add up to the expected total volume."
    
#     for i in range(len(cells)):
#         print(cells[i])
#     exit(-1)
    return np.array(cellsX),np.array(cellsY),np.array(cellsZ),np.array(cells)

def reconstructBaseMesh_distributed(XL,YL,ZL,maxSideLength,cellsX,cellsY,cellsZ,verbose=1):
    '''
    Input the domain parameters and a maximum cell size.  Return a list of the minimally refined mesh coordinates.
    
    :param XL:
    :param YL:
    :param ZL:
    :param maxSideLength:
    '''
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    nx = int( -( (-2*XL) // maxSideLength ) )  # give the number of base cells in the x direction
    ny = int( -( (-2*YL) // maxSideLength ) )  # give the number of base cells in the y direction
    nz = int( -( (-2*ZL) // maxSideLength ) )  # give the number of base cells in the z direction
    

    if verbose>0: rprint(rank,'nx, ny, nz: %i, %i, %i' %(nx,ny,nz))
    x = np.linspace(-XL,XL,nx+1)
    dx = x[1]-x[0]
    y = np.linspace(-YL,YL,ny+1)
    dy = y[1]-y[0]
    z = np.linspace(-ZL,ZL,nz+1)
    dz = z[1]-z[0]
    
    cells = []
#     cellsX = []
#     cellsY = []
#     cellsZ = []
    volumes=[]
    idx=0
    
    for i in range(len(cellsX)):
        xl = cellsX[i]-dx/2
        xh = cellsX[i]+dx/2
        
        yl = cellsY[i]-dy/2
        yh = cellsY[i]+dy/2
        
        zl = cellsZ[i]-dz/2
        zh = cellsZ[i]+dz/2
        
        
        cells.append( [xl, xh, yl, yh, zl, zh ] )
        volumes.append( (xh-xl)*(yh-yl)*(zh-zl) )
    
    
    
    if verbose>0: print("After LB:  Rank %i, number of coarse cells: %i, volume = %f" %(rank, len(cells), np.sum(volumes)) )
#     if verbose>0: rprint(rank,"Rank %i, number of coarse cells: %i" %(rank, len(cells)) )
#     if verbose>0: rprint(rank, "Expected volume:   ", (2*XL*2*YL*2*ZL))
#     if verbose>0: rprint(rank, "Cumulative volume: ", np.sum(volumes))

    totalVolume = comm.allreduce(np.sum(volumes))
    rprint(rank,"After LB:  Total volume: ", totalVolume)
    assert abs((2*XL*2*YL*2*ZL) - totalVolume) < 1e-12, "After LB: base mesh cells volumes do not add up to the expected total volume."
    
#     for i in range(len(cells)):
#         print(cells[i])
#     exit(-1)
    return cells

def buildMeshFromMinimumDepthCells(XL,YL,ZL,maxSideLength,coreRepresentation,inputFile,outputFile,srcdir,order,fine_order,gaugeShift,
                                   divideCriterion,divideParameter1,divideParameter2,divideParameter3,divideParameter4,verbose=0, saveTree=False):
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
            print(atomData)
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
    
    
    cellsX,cellsY,cellsZ,cells=inializeBaseMesh_distributed(XL,YL,ZL,maxSideLength)
    
    print("rank %i" %rank, cellsX,cellsY,cellsZ, cells)
    
    preXmean = np.mean(cellsX)
    preYmean = np.mean(cellsY)
    preZmean = np.mean(cellsZ)
    comm.barrier()
    start=MPI.Wtime()
    cellsX,cellsY,cellsZ = loadBalance(cellsX,cellsY,cellsZ,LBMETHOD='RCB')
    end=MPI.Wtime()
    print("RCB balancing took %f seconds." %(end-start))
    comm.barrier()
    
    postXmean = np.mean(cellsX)
    postYmean = np.mean(cellsY)
    postZmean = np.mean(cellsZ)
    
    cells = reconstructBaseMesh_distributed(XL,YL,ZL,maxSideLength,cellsX,cellsY,cellsZ)
    
    
    print("Rank %i, xmean was %f and is now %f " %(rank,preXmean,postXmean))
    print("Rank %i, ymean was %f and is now %f " %(rank,preYmean,postYmean))
    print("Rank %i, zmean was %f and is now %f " %(rank,preZmean,postZmean))

#     print("rank %i" %rank, cellsX,cellsY,cellsZ)
#     exit(-1)
    
    
    x=np.empty(0)
    y=np.empty(0)
    z=np.empty(0)
    w=np.empty(0)
    xf=np.empty(0)
    yf=np.empty(0)
    zf=np.empty(0)
    wf=np.empty(0)
    
    PtsPerCellCoarse=np.empty(0)
    PtsPerCellFine=np.empty(0)
    for i in range(len(cells)):
#         if i%size==rank:
        if True:  # previously there was a check to determine if this cell should be refined by this proc.  Now assume cells have already been decomposed.
#             print("CALLING refineCell ==================================================")
            if saveTree==False:
                X,Y,Z,W,Xf,Yf,Zf,Wf,pointsPerCell_coarse, pointsPerCell_fine, atoms,nPoints,nOrbitals,nElectrons,referenceEigenvalues = refineCell(nElectrons,nOrbitals,atoms,coreRepresentation,cells[i],inputFile,outputFile,srcdir,order,fine_order,gaugeShift,divideCriterion=divideCriterion,
                                                                                        divideParameter1=divideParameter1, divideParameter2=divideParameter2, divideParameter3=divideParameter3, divideParameter4=divideParameter4, saveTree=saveTree)
            elif saveTree==True:
                X,Y,Z,W,Xf,Yf,Zf,Wf,pointsPerCell_coarse, pointsPerCell_fine, atoms,nPoints,nOrbitals,nElectrons,referenceEigenvalues, tree = refineCell(nElectrons,nOrbitals,atoms,coreRepresentation,cells[i],inputFile,outputFile,srcdir,order,fine_order,gaugeShift,divideCriterion=divideCriterion,
                                                                                        divideParameter1=divideParameter1, divideParameter2=divideParameter2, divideParameter3=divideParameter3, divideParameter4=divideParameter4, saveTree=saveTree)
            
            x=np.append(x,X)
            y=np.append(y,Y)
            z=np.append(z,Z)
            w=np.append(w,W)
            PtsPerCellCoarse=np.append(PtsPerCellCoarse,pointsPerCell_coarse)
            
            xf=np.append(xf,Xf)
            yf=np.append(yf,Yf)
            zf=np.append(zf,Zf)
            wf=np.append(wf,Wf)
            PtsPerCellFine=np.append(PtsPerCellFine,pointsPerCell_fine)
    
    print("rank %i, number of points %i" %(rank,len(x)))
#     comm.barrier()
#     exit(-1)
    ## BUG HERE WHEN RANKS=6.  Need every rank to refine at least one cell.
#     nPoints=len(x)
    if saveTree==False:
        return x,y,z,w,xf,yf,zf,wf,PtsPerCellCoarse, PtsPerCellFine, atoms,PSPs,nPoints,nOrbitals,nElectrons,referenceEigenvalues
    elif saveTree==True:
        return x,y,z,w,xf,yf,zf,wf,PtsPerCellCoarse, PtsPerCellFine, atoms,PSPs,nPoints,nOrbitals,nElectrons,referenceEigenvalues, tree

def func(X,Y,Z,pow):
    
    return X**pow

def integral_func(X,Y,Z,pow,xL,xH,yL,yH,zL,zH):

    return (xH**(pow+1)/(pow+1) - (xL)**(pow+1)/(pow+1))*(yH-yL)*(zH-zL)


def refineCell(nElectrons,nOrbitals,atoms,coreRepresentation,coordinates,inputFile,outputFile,srcdir,order,fine_order,gaugeShift,additionalDepthAtAtoms=0,minDepth=0,divideCriterion='ParentChildrenIntegral',divideParameter1=0,divideParameter2=0,divideParameter3=0,divideParameter4=0, verbose=0, saveTree=False):
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
                coordinateFile=srcdir+coordinateFile, inputFile=srcdir+inputFile, fine_order=fine_order)#, iterationOutFile=outputFile)

    tree.finalDivideBasedOnNuclei(coordinateFile)
   
    tree.buildTree( initializationType='atomic',divideCriterion=divideCriterion, 
                    divideParameter1=divideParameter1, divideParameter2=divideParameter2, divideParameter3=divideParameter3, divideParameter4=divideParameter4, 
                    savedMesh=savedMesh, restart=restart, printTreeProperties=False,onlyFillOne=False)
    
#     tree.finalDivideBasedOnNuclei(coordinateFile)
    
#     tree.exportGridpoints
    X, Y, Z, W, Xf, Yf, Zf, Wf, pointsPerCell_coarse, pointsPerCell_fine, RHO, XV, YV, ZV, vertexIdx, centerIdx, ghostCells = tree.extractXYZ()

    pointsPerCell_coarse = pointsPerCell_coarse.astype(int)
    pointsPerCell_fine = pointsPerCell_fine.astype(int)

#     print("pointsPerCell_coarse = ", pointsPerCell_coarse)
#     print("pointsPerCell_fine   = ", pointsPerCell_fine)
#     print(len(X))
#     print(len(Xf))
#     
#     if not np.array_equal(pointsPerCell_coarse,pointsPerCell_fine):
#         exit(-1)
    
    atoms = tree.atoms
    nPoints = len(X)
    volume = (xmax-xmin)*(ymax-ymin) * (zmax-zmin)
    assert len(X)==len(Y), "len(X) not equal to len(Y) for a base mesh tree"
    assert len(X)==len(Z), "len(X) not equal to len(Z) for a base mesh tree"
    assert len(X)==len(W), "len(X) not equal to len(W) for a base mesh tree"
    assert len(X)==len(RHO), "len(X) not equal to len(RHO) for a base mesh tree"
#     print("sum of weights = ", np.sum(W))
#     print("volume = ",volume)
#     print(W)
    assert abs( np.sum(W) - volume)/volume<1e-12, "Sum of weights doesn't equal base mesh volume."
    assert nPoints>0, "nPoints not > 0 for one of the base mesh trees. "
    
    pow=2
    testfunc = func(X,Y,Z,pow)
    testfuncFine = func(Xf,Yf,Zf,pow)
    computedIntegral = np.sum(testfunc*W)
    computedIntegralFine = np.sum(testfuncFine*Wf)
    analyticIntegral = integral_func(X,Y,Z,pow,xmin,xmax,ymin,ymax,zmin,zmax)
    print(nPoints)
    print("Computed Integral over coarse base mesh:          %f, %1.3e" %(computedIntegral, (analyticIntegral-computedIntegral)/analyticIntegral))
    print("Computed Integral over fine base mesh:            %f, %1.3e" %(computedIntegralFine,(analyticIntegral-computedIntegralFine)/analyticIntegral))
    print("Analytic Integral over base mesh:                   ", analyticIntegral)
    try: 
        assert abs(computedIntegral-analyticIntegral)<1e-8, "Computed (%f) and analytic (%f) integrals not matching for base mesh spanning %f,%f x %f,%f x %f,%f" %(computedIntegral, analyticIntegral, xmin,xmax,ymin,ymax,zmin,zmax)
    except AssertionError as error:
        print(error)
        print(X)
        print(Y)
        print(Z)
        print(W)
        print(np.sum(W))
        exit(-1)

    if saveTree==False:
        tree=None
        return X, Y, Z, W, Xf, Yf, Zf, Wf, pointsPerCell_coarse, pointsPerCell_fine, atoms,nPoints,nOrbitals,nElectrons,referenceEigenvalues
    elif saveTree==True:
        return X, Y, Z, W, Xf, Yf, Zf, Wf, pointsPerCell_coarse, pointsPerCell_fine, atoms,nPoints,nOrbitals,nElectrons,referenceEigenvalues, tree
    else:
        print("What should saveTree be set to?")
        exit(-1)


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
    