import numpy as np
import ctypes
from mpi4py import MPI

from mpiUtilities import rprint


try: 
    _cpu_orthogonalizationRoutines = ctypes.CDLL('libOrthogonalization_cpu.so')
except OSError as e:
    print(e)
    exit(-1)
try: 
    _gpu_orthogonalizationRoutines = ctypes.CDLL('libOrthogonalization_gpu.so')
except OSError as e:
    print(e)
    exit(-1)
        

        
""" Set argtypes of the wrappers. """

try:
#     _cpu_orthogonalizationRoutines.modifiedGramSchmidt_singleWavefunction.argtypes = (np.ctypeslib.ndpointer(dtype=np.intp), 
    _cpu_orthogonalizationRoutines.modifiedGramSchmidt_singleWavefunction.argtypes = (ctypes.POINTER(ctypes.c_double), 
                                                                                      ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int, ctypes.c_int) 
except NameError:
    print("Could not set argtypes of _cpu_orthogonalizationRoutines")
    
try:
#     _gpu_orthogonalizationRoutines.modifiedGramSchmidt_singleWavefunction.argtypes = (np.ctypeslib.ndpointer(dtype=np.intp), 
    _gpu_orthogonalizationRoutines.modifiedGramSchmidt_singleWavefunction.argtypes = (ctypes.POINTER(ctypes.c_double), 
                                                                                      ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int, ctypes.c_int) 
except NameError:
    print("Could not set argtypes of _gpu_orthogonalizationRoutines")




def callOrthogonalization(wavefunctions, target, W, targetWavefunction, gpuPresent):

    numWavefunctions,numPoints = np.shape(wavefunctions)
    print("numWavefunctions,numPoints = ", numWavefunctions,numPoints)
    c_double_p = ctypes.POINTER(ctypes.c_double)    # standard pointer to array of doubles
#     c_double_pp = ctypes.POINTER(ctypes.c_double)   # double pointer (for 2D arrays of doubles)
    
    c_double_pp = np.ctypeslib.ndpointer(dtype=np.float64,
            ndim=2,
            flags='C_CONTIGUOUS'
            )
    
#     wavefunctions_pp = wavefunctions.data_as(c_double_pp)
    target_p = target.ctypes.data_as(c_double_p)
    W_p = W.ctypes.data_as(c_double_p)
    wavefunctions_p = wavefunctions.ctypes.data_as(c_double_p)
    
#     wavefunctions_pp = (wavefunctions.__array_interface__['data'][0] + np.arange(wavefunctions.shape[0])*wavefunctions.strides[0]).astype(np.intp)
#     print("\n\n Inside callOrthogonalization, wavefunctions_pp = ",wavefunctions_pp,"\n\n")
#     doublepp = np.ctypeslib.ndpointer(dtype=np.intp)
    
    
    
    print("Wrapper is now calling modifiedGramSchmidt_singleWavefunction.")
    if gpuPresent==False:
        _cpu_orthogonalizationRoutines.modifiedGramSchmidt_singleWavefunction(wavefunctions_p, target_p, W_p, targetWavefunction, numPoints, numWavefunctions ) 
    elif gpuPresent==True:
        _gpu_orthogonalizationRoutines.modifiedGramSchmidt_singleWavefunction(wavefunctions_p, target_p, W_p, targetWavefunction, numPoints, numWavefunctions ) 

 
    
    return 
#     return wavefunctions
