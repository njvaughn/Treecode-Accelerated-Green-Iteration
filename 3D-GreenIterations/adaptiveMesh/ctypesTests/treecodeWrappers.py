import numpy as np
import ctypes
import time


#  Suppose there is a compiled C file treecode.c function with the header:
#
#
#  double treeEval_C(   int numPars, 
#                      double *sourceX, double *sourceY, double *sourceZ, double *sourceVal,
#                      double *targetX, double *targetY, double *targetZ, double *targetVal){
#                ...
#                ...
#                ...
#                return resultArray 
#  }
#
#

# Compiled with something like:
# gcc -fPIC -shared -o libtreecode.so treecode.c


# _treecodeRoutines = ctypes.CDLL('/Users/nathanvaughn/Documents/GitHub/hybrid-gpu-treecode/lib/libtreedriverWrapper.so')
_treecodeRoutines = ctypes.CDLL('/home/njvaughn/hybrid-gpu-treecode/lib/libtreedriverWrapper.so')

_treecodeRoutines.treedriverWrapper.argtypes = ( ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_double, 
        ctypes.c_int, ctypes.c_double,  ctypes.c_int, ctypes.c_int )

# void treedriverWrapper(int numTargets, int numSources,
#         double *targetX, double *targetY, double *targetZ, double *targetValue,
#         double *sourceX, double *sourceY, double *sourceZ, double *sourceValue, double *sourceWeight,
#         double *outputArray, int pot_type, double kappa,
#         int order, double theta, int maxparnode, int batch_size) {




def callTreedriver(numTargets, numSources, 
                   targetX, targetY, targetZ, targetValue, 
                   sourceX, sourceY, sourceZ, sourceValue, sourceWeight,
                   potentialType, kappa, order, theta, maxParNode, batchSize):

   
    global _treecodeRoutines
    c_double_p = ctypes.POINTER(ctypes.c_double)

    
    targetX_p = targetX.ctypes.data_as(c_double_p)
    targetY_p = targetY.ctypes.data_as(c_double_p)
    targetZ_p = targetZ.ctypes.data_as(c_double_p)
    targetValue_p = targetValue.ctypes.data_as(c_double_p)
    
    
    sourceX_p = sourceX.ctypes.data_as(c_double_p)
    sourceY_p = sourceY.ctypes.data_as(c_double_p)
    sourceZ_p = sourceZ.ctypes.data_as(c_double_p)
    sourceValue_p = sourceValue.ctypes.data_as(c_double_p)
    sourceWeight_p = sourceWeight.ctypes.data_as(c_double_p)
    
    resultArray = np.zeros(numTargets)
    resultArray_p = resultArray.ctypes.data_as(c_double_p)

    _treecodeRoutines.treedriverWrapper(ctypes.c_int(numTargets), ctypes.c_int(numSources),
                                                 targetX_p, targetY_p, targetZ_p, targetValue_p,
                                                 sourceX_p, sourceY_p, sourceZ_p, sourceValue_p, sourceWeight_p,
                                                 resultArray_p, ctypes.c_int(potentialType), ctypes.c_double(kappa),
                                                 ctypes.c_int(order), ctypes.c_double(theta), ctypes.c_int(maxParNode),  ctypes.c_int(batchSize))
    
# void treedriverWrapper(int numTargets, int numSources,
#     double *targetX, double *targetY, double *targetZ, double *targetValue,
#     double *sourceX, double *sourceY, double *sourceZ, double *sourceValue, double *sourceWeight,
#     double *outputArray, int pot_type, double kappa,
#     int order, double theta, int maxparnode, int batch_size) {

    
    return resultArray
