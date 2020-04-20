import sys
import mpi4py.MPI as MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
import numpy as np
import gc


from mpiUtilities import rprint, global_dot
from loadBalancer import loadBalance_manual
sys.path.append('../dataStructures')
from TreeStruct_CC import Tree
from AtomStruct import Atom
from zoltan_wrapper import callZoltan
from meshUtilities import ChebyshevPointsFirstKind,unscaledWeightsFirstKind,weights3DFirstKind



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
#         rprint(0, cells[i])
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
    
    if verbose>0: rprint(0, "Rank %i, number of coarse cells: %i, volume = %f" %(rank, len(cells), np.sum(volumes)) )
#     if verbose>0: rprint(rank,"Rank %i, number of coarse cells: %i" %(rank, len(cells)) )
#     if verbose>0: rprint(rank, "Expected volume:   ", (2*XL*2*YL*2*ZL))
#     if verbose>0: rprint(rank, "Cumulative volume: ", np.sum(volumes))

    totalVolume = comm.allreduce(np.sum(volumes))
    rprint(rank,"Total volume: ", totalVolume)
    assert abs((2*XL*2*YL*2*ZL) - totalVolume) < 1e-12, "base mesh cells volumes do not add up to the expected total volume."
    
#     for i in range(len(cells)):
#         rprint(0, cells[i])
#     exit(-1)
    return np.array(cellsX),np.array(cellsY),np.array(cellsZ),np.array(cells)

def inializeBaseMesh_distributed_duplicated(XL,YL,ZL,maxSideLength,verbose=1): 
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
                
#                 if idx%size==rank:
                cells.append( [x[i], x[i+1], y[j], y[j+1], z[k], z[k+1]] )
                volumes.append( (x[i+1]-x[i])*(y[j+1]-y[j])*(z[k+1]-z[k]) )
                
                cellsX.append( (x[i+1]+x[i])/2 )
                cellsY.append( (y[j+1]+y[j])/2 )
                cellsZ.append( (z[k+1]+z[k])/2 )
                
                idx+=1
    
    if verbose>0: rprint(0, "Rank %i, number of coarse cells: %i, volume = %f" %(rank, len(cells), np.sum(volumes)) )
#     if verbose>0: rprint(rank,"Rank %i, number of coarse cells: %i" %(rank, len(cells)) )
#     if verbose>0: rprint(rank, "Expected volume:   ", (2*XL*2*YL*2*ZL))
#     if verbose>0: rprint(rank, "Cumulative volume: ", np.sum(volumes))

    totalVolume = comm.allreduce(np.sum(volumes))
    rprint(rank,"Total volume: ", totalVolume)
    assert abs((2*XL*2*YL*2*ZL) - totalVolume/size) < 1e-12, "base mesh cells volumes do not add up to the expected total volume."
    
#     for i in range(len(cells)):
#         rprint(0, cells[i])
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
    
    rprint(0, "Rank %i, length of cells arrays: " %rank, len(cellsX))
    
    for i in range(len(cellsX)):
        xl = cellsX[i]-dx/2
        xh = cellsX[i]+dx/2
        
        yl = cellsY[i]-dy/2
        yh = cellsY[i]+dy/2
        
        zl = cellsZ[i]-dz/2
        zh = cellsZ[i]+dz/2
        
        
        cells.append( [xl, xh, yl, yh, zl, zh ] )
        volumes.append( (xh-xl)*(yh-yl)*(zh-zl) )
    
    
    
    if verbose>0: rprint(0, "After LB:  Rank %i, number of coarse cells: %i, volume = %f" %(rank, len(cells), np.sum(volumes)) )
#     if verbose>0: rprint(rank,"Rank %i, number of coarse cells: %i" %(rank, len(cells)) )
#     if verbose>0: rprint(rank, "Expected volume:   ", (2*XL*2*YL*2*ZL))
#     if verbose>0: rprint(rank, "Cumulative volume: ", np.sum(volumes))

    totalVolume = comm.allreduce(np.sum(volumes))
    rprint(rank,"After LB:  Total volume: ", totalVolume)
    assert abs((2*XL*2*YL*2*ZL) - totalVolume) < 1e-12, "After LB: base mesh cells volumes do not add up to the expected total volume."
    
#     for i in range(len(cells)):
#         rprint(0, cells[i])
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
            rprint(0, atomData)
            atom = Atom(atomData[i,0],atomData[i,1],atomData[i,2],atomData[i,3],atomData[i,4],coreRepresentation)
            atoms[i] = atom
            nOrbitals += int(atomData[i,4])
            if coreRepresentation=="AllElectron":
                nElectrons+=atomData[i,3]
            elif coreRepresentation=="Pseudopotential":
                atom.setPseudopotentialObject(PSPs)
                nElectrons += atom.PSP.psp['header']['z_valence']
            else:
                rprint(rank,"What is coreRepresentation?")
                exit(-1)
    
    
    
    

    
    
#     nOrbitals = int( np.ceil(nElectrons/2)*1.2  )   # start with the minimum number of orbitals 

    occupations = 2*np.ones(nOrbitals)

    if verbose>0: rprint(rank,[coordinateFile, outputFile, nElectrons, nOrbitals, 
                          Etotal, Eexchange, Ecorrelation, Eband, gaugeShift])
    
    
    cellsX,cellsY,cellsZ,cells=inializeBaseMesh_distributed(XL,YL,ZL,maxSideLength)  # rank p of P get's every Pth cell.
#     cellsX,cellsY,cellsZ,cells=inializeBaseMesh_distributed_duplicated(XL,YL,ZL,maxSideLength)
    
#     rprint(0, "rank %i" %rank, cellsX,cellsY,cellsZ, cells)
    rprint(0, "rank %i, number of cells before refinement = %i" %(rank,len(cellsX)))

    
#     preXmean = np.mean(cellsX)
#     preYmean = np.mean(cellsY)
#     preZmean = np.mean(cellsZ)
#     comm.barrier()
#     start=MPI.Wtime()
# #     cellsX,cellsY,cellsZ = loadBalance(cellsX,cellsY,cellsZ,LBMETHOD='RCB')
#     cellsX,cellsY,cellsZ = loadBalance_manual(cellsX,cellsY,cellsZ)
#     end=MPI.Wtime()
#     rprint(0, "Manual RCB balancing took %f seconds and resulted in %i cells for rank %i." %((end-start),len(cellsX),rank))
    comm.barrier()
    
#     postXmean = np.mean(cellsX)
#     postYmean = np.mean(cellsY)
#     postZmean = np.mean(cellsZ)
#     
#     cells = reconstructBaseMesh_distributed(XL,YL,ZL,maxSideLength,cellsX,cellsY,cellsZ)
#     
#     
#     rprint(0, "Rank %i, xmean was %f and is now %f " %(rank,preXmean,postXmean))
#     rprint(0, "Rank %i, ymean was %f and is now %f " %(rank,preYmean,postYmean))
#     rprint(0, "Rank %i, zmean was %f and is now %f " %(rank,preZmean,postZmean))
    
#     exit(-1)

#     rprint(0, "rank %i" %rank, cellsX,cellsY,cellsZ)
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
#     
#     ## This is being replaced to refine and balance cells (not quadrature points)
#     for i in range(len(cells)):
#         if i%size==rank:
# #         if True:  # previously there was a check to determine if this cell should be refined by this proc.  Now assume cells have already been decomposed.
# #             rprint(0, "CALLING refineCell ==================================================")
#             if saveTree==False:
#                 X,Y,Z,W,Xf,Yf,Zf,Wf,pointsPerCell_coarse, pointsPerCell_fine, atoms,nPoints,nOrbitals,nElectrons,referenceEigenvalues = refineCell(nElectrons,nOrbitals,atoms,coreRepresentation,cells[i],inputFile,outputFile,srcdir,order,fine_order,gaugeShift,divideCriterion=divideCriterion,
#                                                                                         divideParameter1=divideParameter1, divideParameter2=divideParameter2, divideParameter3=divideParameter3, divideParameter4=divideParameter4, saveTree=saveTree)
#             elif saveTree==True:
#                 X,Y,Z,W,Xf,Yf,Zf,Wf,pointsPerCell_coarse, pointsPerCell_fine, atoms,nPoints,nOrbitals,nElectrons,referenceEigenvalues, tree = refineCell(nElectrons,nOrbitals,atoms,coreRepresentation,cells[i],inputFile,outputFile,srcdir,order,fine_order,gaugeShift,divideCriterion=divideCriterion,
#                                                                                         divideParameter1=divideParameter1, divideParameter2=divideParameter2, divideParameter3=divideParameter3, divideParameter4=divideParameter4, saveTree=saveTree)
#             
#             x=np.append(x,X)
#             y=np.append(y,Y)
#             z=np.append(z,Z)
#             w=np.append(w,W)
#             PtsPerCellCoarse=np.append(PtsPerCellCoarse,pointsPerCell_coarse)
#             
#             xf=np.append(xf,Xf)
#             yf=np.append(yf,Yf)
#             zf=np.append(zf,Zf)
#             wf=np.append(wf,Wf)
#             PtsPerCellFine=np.append(PtsPerCellFine,pointsPerCell_fine)


    ### New Scheme that respects cells when decomposing ###
    
    # Step 1: Generate list of coarse cells
    # Step 2: Let each processor refine some of the coarse cells.  Return list of cells, not quadrature points
    # Step 3: Load balance the cells
    # Step 4: Have each processor generate the mesh of quadrature points on its list of cells. 
    # Step 5: Build two level mesh for an cells meeting that criteria.
            
            
    refinedCellsX=np.empty(0)
    refinedCellsY=np.empty(0)
    refinedCellsZ=np.empty(0)

    refinedCellsDX=np.empty(0)
    refinedCellsDY=np.empty(0)
    refinedCellsDZ=np.empty(0)

    refinedPtsPerCellCoarse=np.empty(0,dtype=np.int32)
    refinedPtsPerCellFine=np.empty(0,dtype=np.int32)
    
    divideBasedOnNuclei=False
    rprint(rank, "Not dividing cells at nuclei. Seems to be okay for pseudopoential mesh.")
    
    for i in range(len(cells)):
#         if i%size==rank:
        if True:  ## the i%size==rank check already occurs when constructing the set of cells.
            if saveTree==False:
                cellsX,cellsY,cellsZ,cellsDX,cellsDY,cellsDZ,PtsPerCellCoarse,PtsPerCellFine, atoms,nOrbitals,nElectrons,referenceEigenvalues = refineCellReturnCells(nElectrons,nOrbitals,atoms,coreRepresentation,cells[i],inputFile,outputFile,srcdir,order,fine_order,gaugeShift,divideCriterion=divideCriterion,
                                                                                        divideParameter1=divideParameter1, divideParameter2=divideParameter2, divideParameter3=divideParameter3, divideParameter4=divideParameter4, saveTree=saveTree,divideBasedOnNuclei=divideBasedOnNuclei)
            elif saveTree==True:
                cellsX,cellsY,cellsZ,cellsDX,cellsDY,cellsDZ,PtsPerCellCoarse,PtsPerCellFine, atoms,nPoints,nOrbitals,nElectrons,referenceEigenvalues, tree = refineCellReturnCells(nElectrons,nOrbitals,atoms,coreRepresentation,cells[i],inputFile,outputFile,srcdir,order,fine_order,gaugeShift,divideCriterion=divideCriterion,
                                                                                        divideParameter1=divideParameter1, divideParameter2=divideParameter2, divideParameter3=divideParameter3, divideParameter4=divideParameter4, saveTree=saveTree,divideBasedOnNuclei=divideBasedOnNuclei)
            
            refinedCellsX=np.append(refinedCellsX,cellsX)
            refinedCellsY=np.append(refinedCellsY,cellsY)
            refinedCellsZ=np.append(refinedCellsZ,cellsZ)
            refinedCellsDX=np.append(refinedCellsDX,cellsDX)
            refinedCellsDY=np.append(refinedCellsDY,cellsDY)
            refinedCellsDZ=np.append(refinedCellsDX,cellsDZ)
            refinedPtsPerCellCoarse=np.append(refinedPtsPerCellCoarse,PtsPerCellCoarse)
            refinedPtsPerCellFine=np.append(refinedPtsPerCellFine,PtsPerCellFine)
    
            del cellsX
            del cellsY
            del cellsZ
            del cellsDX
            del cellsDY
            del cellsDZ   
            del PtsPerCellCoarse
            del PtsPerCellFine
    
    # compute numCellsLocal and globalStart
    numCellsLocal=len(refinedCellsX)
    cellsPerRank = np.zeros(size,dtype=np.int32)
    cellsPerRank[rank]=numCellsLocal
    
    cellsPerRank=comm.allreduce(cellsPerRank)
    globalStart=0
    for i in range(rank):
        globalStart+=cellsPerRank[i]
    
    refinedPtsPerCellCoarse = refinedPtsPerCellCoarse.astype(np.int32)
    refinedPtsPerCellFine = refinedPtsPerCellFine.astype(np.int32)
    if verbose>0:
        rprint(0, "rank %i, number of cells after refinement = %i, starts at %i" %(rank,numCellsLocal, globalStart))
        rprint(0, "max, min of refinedPtsPerCellCoarse: ", max(refinedPtsPerCellCoarse), min(refinedPtsPerCellCoarse))
        rprint(0, "max, min of refinedPtsPerCellFine: ", max(refinedPtsPerCellFine), min(refinedPtsPerCellFine))
        rprint(0, "dtype of refinedPtsPerCellCoarse: ", refinedPtsPerCellCoarse.dtype)
    comm.barrier()
    ## Check that the cell decompositions make sense before load balancing.
    localVolume = 0
    for i in range(len(refinedCellsX)):
        localVolume += (refinedCellsDX[i]*refinedCellsDY[i]*refinedCellsDZ[i])
    rprint(0, "rank %i local volume: %f" %(rank,localVolume))
    totalVolume = comm.allreduce(np.sum(localVolume))
    rprint(rank,"Total volume: ", totalVolume)
    assert abs((2*XL*2*YL*2*ZL) - totalVolume) < 1e-12, "BEFORE LOAD BALANCING: base mesh cells volumes do not add up to the expected total volume.  Expected volume = %f" %(2*XL*2*YL*2*ZL) 
      
    
        

    
    
    ## Load balance the cells
    cellsX,cellsY,cellsZ,cellsDX,cellsDY,cellsDZ,PtsPerCellCoarse,PtsPerCellFine, newNumCells = callZoltan(refinedCellsX,refinedCellsY,refinedCellsZ,refinedCellsDX,refinedCellsDY,refinedCellsDZ, refinedPtsPerCellCoarse, refinedPtsPerCellFine, numCellsLocal, globalStart)
    rprint(0, "rank %i, number of cells after balancing = %i" %(rank,len(cellsX)))
    comm.barrier()
#     gc.collect()
#     comm.barrier()
#     rprint(0, "Called garbage collector after calling Zoltan.")
    
    ## Check that the cell decompositions make sense after load balancing.
    localVolume = 0
    for i in range(len(cellsX)):
        localVolume += (cellsDX[i]*cellsDY[i]*cellsDZ[i])
    
    rprint(0, "rank %i, local volume after refinement: %f" %(rank,localVolume))
    totalVolume = comm.allreduce(np.sum(localVolume))
    
    rprint(rank,"Total volume: ", totalVolume)
    assert abs((2*XL*2*YL*2*ZL) - totalVolume) < 1e-12, "AFTER LOAD BALANCING: base mesh cells volumes do not add up to the expected total volume.  Expected volume = %f" %(2*XL*2*YL*2*ZL)
     
     
    
    for i in range(len(cellsX)):
        # construct quadrature points for base mesh
        xl=cellsX[i]-cellsDX[i]/2
        xh=cellsX[i]+cellsDX[i]/2
        
        yl=cellsY[i]-cellsDY[i]/2
        yh=cellsY[i]+cellsDY[i]/2
        
        zl=cellsZ[i]-cellsDZ[i]/2
        zh=cellsZ[i]+cellsDZ[i]/2
        
        xc,yc,zc,wc = coarseQuadraturePointsSingleCell(xl,xh,yl,yh,zl,zh,order)
        
        x = np.append(x,xc)
        y = np.append(y,yc)
        z = np.append(z,zc)
        w = np.append(w,wc)
        
        # construct quadrature points for two-level mesh
        if PtsPerCellFine[i]==PtsPerCellCoarse[i]:
            # copy same set of points
            xfc=np.copy(xc)
            yfc=np.copy(yc)
            zfc=np.copy(zc)
            wfc=np.copy(wc)
        else:
            # generate the refined set of PtsPerCellFine[i] points
            xfc,yfc,zfc,wfc = fineQuadraturePointsSingleCell(xl,xh,yl,yh,zl,zh,order,PtsPerCellFine[i])
        
        xf = np.append(xf,xfc)
        yf = np.append(yf,yfc)
        zf = np.append(zf,zfc)
        wf = np.append(wf,wfc)
        
        
    
    comm.barrier()
    nPoints=len(x)
    PtsPerCellCoarse=np.array(PtsPerCellCoarse, dtype=np.int32)
    PtsPerCellFine=np.array(PtsPerCellFine, dtype=np.int32)
    if verbose>0:
        rprint(0, "len(x) = ", len(x))
        rprint(0, "first few PtsPerCellCoarse: ", PtsPerCellCoarse[0:5])
        rprint(0, "np.sum(PtsPerCellCoarse) = ", np.sum(PtsPerCellCoarse))
        rprint(0, "max, min of PtsPerCellCoarse: ", max(PtsPerCellCoarse), min(PtsPerCellCoarse))
    assert len(x) == int(np.sum(PtsPerCellCoarse)), "Error: len(x) != np.sum(PtsPerCellCoarse)"
    assert len(xf) == np.sum(PtsPerCellFine), "Error: len(xf) != np.sum(PtsPerCellFine)"

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
    if verbose>0: rprint(0, xmin,xmax,ymin,ymax,zmin,zmax)

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
#     rprint(0, "Not dividing based on Nuclei.")
    tree.buildTree( initializationType='atomic',divideCriterion=divideCriterion, 
                    divideParameter1=divideParameter1, divideParameter2=divideParameter2, divideParameter3=divideParameter3, divideParameter4=divideParameter4, 
                    savedMesh=savedMesh, restart=restart, printTreeProperties=False,onlyFillOne=False)
    
#     tree.finalDivideBasedOnNuclei(coordinateFile)
    
#     tree.exportGridpoints
    X, Y, Z, W, Xf, Yf, Zf, Wf, pointsPerCell_coarse, pointsPerCell_fine, RHO, XV, YV, ZV, vertexIdx, centerIdx, ghostCells = tree.extractXYZ()

    pointsPerCell_coarse = pointsPerCell_coarse.astype(int)
    pointsPerCell_fine = pointsPerCell_fine.astype(int)

#     rprint(0, "pointsPerCell_coarse = ", pointsPerCell_coarse)
#     rprint(0, "pointsPerCell_fine   = ", pointsPerCell_fine)
#     rprint(0, len(X))
#     rprint(0, len(Xf))
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
#     rprint(0, "sum of weights = ", np.sum(W))
#     rprint(0, "volume = ",volume)
#     rprint(0, W)
    assert abs( np.sum(W) - volume)/volume<1e-12, "Sum of weights doesn't equal base mesh volume."
    assert nPoints>0, "nPoints not > 0 for one of the base mesh trees. "
    
    pow=2
    testfunc = func(X,Y,Z,pow)
    testfuncFine = func(Xf,Yf,Zf,pow)
    computedIntegral = np.sum(testfunc*W)
    computedIntegralFine = np.sum(testfuncFine*Wf)
    analyticIntegral = integral_func(X,Y,Z,pow,xmin,xmax,ymin,ymax,zmin,zmax)
    if verbose>0:
        rprint(0, nPoints)
        rprint(0, "Computed Integral over coarse base mesh:          %f, %1.3e" %(computedIntegral, (analyticIntegral-computedIntegral)/analyticIntegral))
        rprint(0, "Computed Integral over fine base mesh:            %f, %1.3e" %(computedIntegralFine,(analyticIntegral-computedIntegralFine)/analyticIntegral))
        rprint(0, "Analytic Integral over base mesh:                   ", analyticIntegral)
    try: 
        assert abs(computedIntegral-analyticIntegral)<1e-8, "Computed (%f) and analytic (%f) integrals not matching for base mesh spanning %f,%f x %f,%f x %f,%f" %(computedIntegral, analyticIntegral, xmin,xmax,ymin,ymax,zmin,zmax)
    except AssertionError as error:
        rprint(0, error)
        rprint(0, X)
        rprint(0, Y)
        rprint(0, Z)
        rprint(0, W)
        rprint(0, np.sum(W))
        exit(-1)

    if saveTree==False:
        tree=None
        return X, Y, Z, W, Xf, Yf, Zf, Wf, pointsPerCell_coarse, pointsPerCell_fine, atoms,nPoints,nOrbitals,nElectrons,referenceEigenvalues
    elif saveTree==True:
        return X, Y, Z, W, Xf, Yf, Zf, Wf, pointsPerCell_coarse, pointsPerCell_fine, atoms,nPoints,nOrbitals,nElectrons,referenceEigenvalues, tree
    else:
        rprint(0, "What should saveTree be set to?")
        exit(-1)
        
def refineCellReturnCells(nElectrons,nOrbitals,atoms,coreRepresentation,coordinates,inputFile,outputFile,srcdir,order,fine_order,gaugeShift,additionalDepthAtAtoms=0,minDepth=0,divideCriterion='ParentChildrenIntegral',divideParameter1=0,divideParameter2=0,divideParameter3=0,divideParameter4=0, verbose=0, 
                          saveTree=False,divideBasedOnNuclei=False):
    '''
    setUp() gets called before every test below.
    '''
    [xmin, xmax, ymin, ymax, zmin, zmax] = coordinates
    if verbose>0: rprint(0, xmin,xmax,ymin,ymax,zmin,zmax)

    [coordinateFile, referenceEigenvaluesFile, DummyOutputFile] = np.genfromtxt(inputFile,dtype="|U100")[:3]
    [Eband, Ekinetic, Eexchange, Ecorrelation, Eelectrostatic, Etotal] = np.genfromtxt(inputFile)[3:]
    
    
    savedMesh=''
    restart=False
    referenceEigenvalues = np.array( np.genfromtxt(srcdir+referenceEigenvaluesFile,delimiter=',',dtype=float) )
    if verbose>0: rprint(rank,referenceEigenvalues)
    if verbose>0: rprint(rank,np.shape(referenceEigenvalues))
    tree = Tree(xmin,xmax,order,ymin,ymax,order,zmin,zmax,order,atoms,coreRepresentation,nElectrons,nOrbitals,additionalDepthAtAtoms=additionalDepthAtAtoms,minDepth=minDepth,gaugeShift=gaugeShift,
                coordinateFile=srcdir+coordinateFile, inputFile=srcdir+inputFile, fine_order=fine_order)#, iterationOutFile=outputFile)

    if divideBasedOnNuclei:
        tree.finalDivideBasedOnNuclei(coordinateFile)
#     rprint(rank,"Not dividing based on Nuclei.")
    tree.buildTree( initializationType='atomic',divideCriterion=divideCriterion, 
                    divideParameter1=divideParameter1, divideParameter2=divideParameter2, divideParameter3=divideParameter3, divideParameter4=divideParameter4, 
                    savedMesh=savedMesh, restart=restart, printTreeProperties=False,onlyFillOne=False)
    
#     tree.finalDivideBasedOnNuclei(coordinateFile)
    
#     tree.exportGridpoints
    cellsX,cellsY,cellsZ,cellsDX,cellsDY,cellsDZ,PtsPerCellCoarse,PtsPerCellFine, RHO, XV, YV, ZV, vertexIdx, centerIdx, ghostCells = tree.extractCellXYZ()
    
    

    PtsPerCellCoarse = PtsPerCellCoarse.astype(int)
    PtsPerCellFine = PtsPerCellFine.astype(int)
    
    atoms = tree.atoms
    

    if saveTree==False:
        tree=None
        return cellsX,cellsY,cellsZ,cellsDX,cellsDY,cellsDZ,PtsPerCellCoarse,PtsPerCellFine, atoms,nOrbitals,nElectrons,referenceEigenvalues
#         return X, Y, Z, W, Xf, Yf, Zf, Wf, pointsPerCell_coarse, pointsPerCell_fine, atoms,nPoints,nOrbitals,nElectrons,referenceEigenvalues
    elif saveTree==True:
        return cellsX,cellsY,cellsZ,cellsDX,cellsDY,cellsDZ,PtsPerCellCoarse,PtsPerCellFine, atoms,nOrbitals,nElectrons,referenceEigenvalues
#         return X, Y, Z, W, Xf, Yf, Zf, Wf, pointsPerCell_coarse, pointsPerCell_fine, atoms,nPoints,nOrbitals,nElectrons,referenceEigenvalues, tree
    else:
        rprint(0, "What should saveTree be set to?")
        exit(-1)


def coarseQuadraturePointsSingleCell(xlow,xhigh,ylow,yhigh,zlow,zhigh,order):
    
    xvec = ChebyshevPointsFirstKind(xlow, xhigh, order)
    yvec = ChebyshevPointsFirstKind(ylow, yhigh, order)
    zvec = ChebyshevPointsFirstKind(zlow, zhigh, order)
    
    W = unscaledWeightsFirstKind(order)  # assumed px=py=pz
    wt = weights3DFirstKind(xlow, xhigh, order, ylow, yhigh, order, zlow, zhigh, order, W)
    
    x=np.zeros((order+1)**3)
    y=np.zeros((order+1)**3)
    z=np.zeros((order+1)**3)
    w=np.zeros((order+1)**3)
    
    idx=0
    for i in range(order+1):
        xt = xvec[i]
        for j in range(order+1):
            yt=yvec[j]
            for k in range(order+1):
                zt=zvec[k]
                
                x[idx] = xt
                y[idx] = yt
                z[idx] = zt
                w[idx] = wt[i][j][k]
                
                idx+=1
    
    
    
    return x,y,z,w


def fineQuadraturePointsSingleCell(xlow,xhigh,ylow,yhigh,zlow,zhigh,order,ptsInFineCell):
    
    # Step 1: determine how many fine chilren are in this cell.
    ptsInCoarseCell=(order+1)**3
    numChildren = int( ptsInFineCell/ptsInCoarseCell )
    
#     rprint(0,"ptsInFineCell = ", ptsInFineCell)
#     rprint(0, "ptsInCoarseCell = ", ptsInCoarseCell)
#     rprint(0, "numChildren = ", numChildren)
    
    if numChildren==1:
        xbounds=[xlow,xhigh]
        ybounds=[ylow,yhigh]
        zbounds=[zlow,zhigh]
    elif numChildren==8:
        xmid=(xlow+xhigh)/2
        ymid=(ylow+yhigh)/2
        zmid=(zlow+zhigh)/2
        
        xbounds=[xlow,xmid,xhigh]
        ybounds=[ylow,ymid,yhigh]
        zbounds=[zlow,zmid,zhigh]
    elif numChildren==27:
        xmidL=(2*xlow+xhigh)/3
        ymidL=(2*ylow+yhigh)/3
        zmidL=(2*zlow+zhigh)/3
        
        xmidR=(2*xhigh+xlow)/3
        ymidR=(2*yhigh+ylow)/3
        zmidR=(2*zhigh+zlow)/3
        
        xbounds=[xlow,xmidL,xmidR,xhigh]
        ybounds=[ylow,ymidL,ymidR,yhigh]
        zbounds=[zlow,zmidL,zmidR,zhigh]
    elif numChildren==64:
        xmidL=(3*xlow+xhigh)/4
        ymidL=(3*ylow+yhigh)/4
        zmidL=(3*zlow+zhigh)/4
        
        xmid=(xlow+xhigh)/2
        ymid=(ylow+yhigh)/2
        zmid=(zlow+zhigh)/2
        
        xmidR=(3*xhigh+xlow)/4
        ymidR=(3*yhigh+ylow)/4
        zmidR=(3*zhigh+zlow)/4
        
        xbounds=[xlow,xmidL,xmid,xmidR,xhigh]
        ybounds=[ylow,ymidL,ymid,ymidR,yhigh]
        zbounds=[zlow,zmidL,zmid,zmidR,zhigh]
    else:
        rprint(0, "Number of children in fine mesh was not 1, 8, 27, or 64.  It was %i.  Exiting." %numChildren)
        exit(-1)
    
    
    x=np.zeros(ptsInFineCell)
    y=np.zeros(ptsInFineCell)
    z=np.zeros(ptsInFineCell)
    w=np.zeros(ptsInFineCell)
    W = unscaledWeightsFirstKind(order)  # assumed px=py=pz
    idx=0
    for ii in range(len(xbounds)-1):
        for jj in range(len(ybounds)-1):
            for kk in range(len(zbounds)-1):
        
                xvec = ChebyshevPointsFirstKind(xbounds[ii], xbounds[ii+1], order)
                yvec = ChebyshevPointsFirstKind(ybounds[jj], ybounds[jj+1], order)
                zvec = ChebyshevPointsFirstKind(zbounds[kk], zbounds[kk+1], order)
                
                w_temp = weights3DFirstKind(xbounds[ii], xbounds[ii+1], order, ybounds[jj], ybounds[jj+1], order, zbounds[kk], zbounds[kk+1], order, W)
#                 rprint(0, "shape of w_temp: ", np.shape(w_temp))
                localIdx=0
                for i in range(order+1):
                    xt = xvec[i]
                    for j in range(order+1):
                        yt=yvec[j]
                        for k in range(order+1):
                            zt=zvec[k]
                            
                            x[idx] = xt
                            y[idx] = yt
                            z[idx] = zt
                            w[idx] = w_temp[i][j][k]
                            
                            idx+=1
                            localIdx+=1
    
    
    assert idx==ptsInFineCell, "idx != ptsInFineCell after constructing fine mesh quadrature points"

    return x,y,z,w


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
    