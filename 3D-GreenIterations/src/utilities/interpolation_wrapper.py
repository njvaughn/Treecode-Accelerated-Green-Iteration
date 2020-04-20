import numpy as np
import ctypes
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
from mpiUtilities import rprint

try: 
    _cpu_interpolationRoutines = ctypes.CDLL('libInterpolation_cpu.so')
except OSError as e:
    rprint(rank, e)
#     exit(-1)
try: 
    _gpu_interpolationRoutines = ctypes.CDLL('libInterpolation_gpu.so')
except OSError:
    rprint(rank, "Warning: Could not load GPU interpolation library.  Ignore if not using GPUs.") 
#     exit(-1)
        

        
""" Set argtypes of the wrappers. """

try:
    _cpu_interpolationRoutines.InterpolateBetweenTwoMeshes.argtypes = ( 
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int), ctypes.c_int,
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int), ctypes.c_int,
            ctypes.c_int,  ctypes.c_int ) 
except NameError:
    rprint(rank, "[interpolationCould not set argtypes of _cpu_interpolationRoutines")
    
try:
    _gpu_interpolationRoutines.InterpolateBetweenTwoMeshes.argtypes = ( 
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int), ctypes.c_int,
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int), ctypes.c_int,
            ctypes.c_int,  ctypes.c_int ) 
except NameError:
    rprint(rank, "Could not set argtypes of _gpu_interpolationRoutines")




def callInterpolator(coarseX, coarseY, coarseZ, coarseF, pointsPerCoarseCell,
                    fineX, fineY, fineZ, pointsPerFineCell, numberOfCells, order,gpuPresent):
        
    '''
    python function which creates pointers to the arrays and calls the compiled C interpolation routines.
    returns the results array.
    '''

    pointsPerCoarseCell = pointsPerCoarseCell.astype(np.int32)
    pointsPerFineCell = pointsPerFineCell.astype(np.int32)
    
#     rprint(rank, pointsPerFineCell[0:5])
#     exit(-1)
   
    c_double_p = ctypes.POINTER(ctypes.c_double)
    c_int_p = ctypes.POINTER(ctypes.c_int)

    
    coarseX_p = coarseX.ctypes.data_as(c_double_p)
    coarseY_p = coarseY.ctypes.data_as(c_double_p)
    coarseZ_p = coarseZ.ctypes.data_as(c_double_p)
    coarseF_p = coarseF.ctypes.data_as(c_double_p)
    pointsPerCoarseCell_p = pointsPerCoarseCell.ctypes.data_as(c_int_p)
    
    
    fineX_p = fineX.ctypes.data_as(c_double_p)
    fineY_p = fineY.ctypes.data_as(c_double_p)
    fineZ_p = fineZ.ctypes.data_as(c_double_p)
    fineF = np.zeros(len(fineX))  
    fineF_p =  fineF.ctypes.data_as(c_double_p)
    pointsPerFineCell_p = pointsPerFineCell.ctypes.data_as(c_int_p)


    if gpuPresent==False:
        _cpu_interpolationRoutines.InterpolateBetweenTwoMeshes(
                                                 coarseX_p, coarseY_p, coarseZ_p, coarseF_p, pointsPerCoarseCell_p, ctypes.c_int(len(coarseX)),
                                                 fineX_p,   fineY_p,   fineZ_p,   fineF_p,   pointsPerFineCell_p,   ctypes.c_int(len(fineX)),
                                                 ctypes.c_int(numberOfCells), 
                                                 ctypes.c_int(order) ) 
    elif gpuPresent==True:
        _gpu_interpolationRoutines.InterpolateBetweenTwoMeshes(
                                                 coarseX_p, coarseY_p, coarseZ_p, coarseF_p, pointsPerCoarseCell_p, ctypes.c_int(len(coarseX)),
                                                 fineX_p,   fineY_p,   fineZ_p,   fineF_p,   pointsPerFineCell_p,   ctypes.c_int(len(fineX)),
                                                 ctypes.c_int(numberOfCells), 
                                                 ctypes.c_int(order) ) 
 
    
    return fineF
