import numpy as np
import ctypes
from mpi4py import MPI



# try:
#     _gpu_treecodeRoutines = ctypes.CDLL('libtreelib-gpu.so')
# except ImportError:
#     print('Unable to import _gpu_treecodeRoutines due to ImportError')
# except OSError:
#     print('Unable to import _gpu_treecodeRoutines due to OSError')
# 
# try:
# #     _cpu_treecodeRoutines = ctypes.CDLL('/home/njvaughn/.local/lib/libtreelib-cpu.so')
#     _cpu_treecodeRoutines = ctypes.CDLL('libtreelib-cpu.dylib')
# except ImportError as e:
#     try:
#         _cpu_treecodeRoutines = ctypes.CDLL('libtreelib-cpu.so')
#     except ImportError:
#         print('Unable to import _cpu_treecodeRoutines due to ImportError')
# #     return
# except OSError as e:
#     try:
#         _cpu_treecodeRoutines = ctypes.CDLL('libtreelib-cpu.so')
#     except OSError:
#         print('Unable to import _cpu_treecodeRoutines due to OSError')

# _cpu_treecodeRoutines = ctypes.CDLL('libtreelib-cpu.dylib')
try: 
    _cpu_treecodeRoutines = ctypes.CDLL('libtreelib-cpu.so')
except OSError:
        _cpu_treecodeRoutines = ctypes.CDLL('libtreelib-cpu.dylib')
    
    
        
# try:
#     _gpu_treecodeRoutines.treedriverWrapper.argtypes = ( ctypes.c_int, ctypes.c_int,
#             ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
#             ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
#             ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_double, 
#             ctypes.c_int, ctypes.c_double,  ctypes.c_int )
# except NameError:
#     pass

try:
    _cpu_treecodeRoutines.treedriverWrapper.argtypes = ( ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_double, 
            ctypes.c_int, ctypes.c_double,  ctypes.c_int,  ctypes.c_int ) 
except NameError:
    pass

print('_treecodeRoutines set.')



def callTreedriver(numTargets, numSources, 
                   targetX, targetY, targetZ, targetValue, 
                   sourceX, sourceY, sourceZ, sourceValue, sourceWeight,
                   potentialType, kappa, order, theta, maxParNode, batchSize, GPUversion):

   
    global _treecodeRoutines
    c_double_p = ctypes.POINTER(ctypes.c_double)

    
    targetX_p = targetX.ctypes.data_as(c_double_p)
    targetY_p = targetY.ctypes.data_as(c_double_p)
    targetZ_p = targetZ.ctypes.data_as(c_double_p)
    targetValue_p = targetValue.ctypes.data_as(c_double_p)
    
    
    sourceX_p = sourceX.ctypes.data_as(c_double_p)
    sourceY_p = sourceY.ctypes.data_as(c_double_p)
    sourceZ_p = sourceZ.ctypes.data_as(c_double_p)  
    sourceValue_p =  sourceValue.ctypes.data_as(c_double_p)
    sourceWeight_p = sourceWeight.ctypes.data_as(c_double_p)
    
    resultArray = np.zeros(numTargets)
    resultArray_p = resultArray.ctypes.data_as(c_double_p)
    
    
    if GPUversion==True:
        _gpu_treecodeRoutines.treedriverWrapper(ctypes.c_int(numTargets),  ctypes.c_int(numSources),
                                                     targetX_p, targetY_p, targetZ_p, targetValue_p,
                                                     sourceX_p, sourceY_p, sourceZ_p, sourceValue_p, sourceWeight_p,
                                                     resultArray_p, ctypes.c_int(potentialType), ctypes.c_double(kappa),
                                                     ctypes.c_int(order), ctypes.c_double(theta), ctypes.c_int(maxParNode), ctypes.c_int(batchSize) )
    elif GPUversion==False: # No gpu present
#         print('No GPU, calling CPU treecode.')
        _cpu_treecodeRoutines.treedriverWrapper(ctypes.c_int(numTargets),  ctypes.c_int(numSources),
                                                     targetX_p, targetY_p, targetZ_p, targetValue_p,
                                                     sourceX_p, sourceY_p, sourceZ_p, sourceValue_p, sourceWeight_p,
                                                     resultArray_p, ctypes.c_int(potentialType), ctypes.c_double(kappa),
                                                     ctypes.c_int(order), ctypes.c_double(theta), ctypes.c_int(maxParNode), ctypes.c_int(batchSize) ) 
#         print('Control returned to python...') 
    else: 
        print("What should GPUversion be set to in the wrapper?")
        return 
    


    return resultArray
