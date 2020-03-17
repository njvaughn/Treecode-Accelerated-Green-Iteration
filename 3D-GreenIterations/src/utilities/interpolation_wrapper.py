import numpy as np
import ctypes
from mpi4py import MPI

from mpiUtilities import rprint


try: 
    _cpu_interpolationRoutines = ctypes.CDLL('libInterpolation_gpu.so')
#     _cpu_interpolationRoutines = ctypes.CDLL('libInterpolation_gpu.so')
except OSError as e:
    print(e)
    exit(-1)
#         _cpu_interpolationRoutines = ctypes.CDLL('libInterpolation.dylib')
        

        
""" Set argtypes of the wrappers. """

try:
    _cpu_interpolationRoutines.InterpolateBetweenTwoMeshes.argtypes = ( 
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int), ctypes.c_int,
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int), ctypes.c_int,
            ctypes.c_int,  ctypes.c_int ) 
except NameError:
    print("Could not set argtypes of _cpu_interpolationRoutines")

print('_treecodeRoutines set.')




def callInterpolator(coarseX, coarseY, coarseZ, coarseF, pointsPerCoarseCell,
                    fineX, fineY, fineZ, pointsPerFineCell, numberOfCells, order):
        
    '''
    python function which creates pointers to the arrays and calls the compiled C interpolation routines.
    returns the results array.
    '''

    pointsPerCoarseCell = pointsPerCoarseCell.astype(np.int32)
    pointsPerFineCell = pointsPerFineCell.astype(np.int32)
    
#     print(pointsPerFineCell[0:5])
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

    _cpu_interpolationRoutines.InterpolateBetweenTwoMeshes(
                                                 coarseX_p, coarseY_p, coarseZ_p, coarseF_p, pointsPerCoarseCell_p, ctypes.c_int(len(coarseX)),
                                                 fineX_p,   fineY_p,   fineZ_p,   fineF_p,   pointsPerFineCell_p,   ctypes.c_int(len(fineX)),
                                                 ctypes.c_int(numberOfCells), 
                                                 ctypes.c_int(order) ) 
 
    
    return fineF