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


# _convolutionRoutines = ctypes.CDLL('./lib/libconvolutionRoutines_noOpenMP.so')
# _convolutionRoutines = ctypes.CDLL('./lib/libconvolutionRoutines.so')
# _convolutionRoutines = ctypes.CDLL('/Users/nathanvaughn/Documents/GitHub/Greens-Functions-Iterative-Methods/3D-GreenIterations/adaptiveMesh/ctypesTests/lib/libconvolutionRoutines.so')
_convolutionRoutines = ctypes.CDLL('../ctypesTests/lib/libconvolutionRoutines.so')
# global rootDirectory
# _convolutionRoutines = ctypes.CDLL(rootDirectory + 'ctypesTests/lib/libconvolutionRoutines.so')
_convolutionRoutines.directSum.argtypes = (ctypes.c_int, ctypes.c_int, 
                              ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                              ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double) )



def treeEval_Python(numTargets, numSources, sourceX, sourceY, sourceZ, sourceW, targetX, targetY, targetZ, targetW):
   
    global _convolutionRoutines
    c_double_p = ctypes.POINTER(ctypes.c_double)

    
    targetX_p = targetX.ctypes.data_as(c_double_p)
    targetY_p = targetY.ctypes.data_as(c_double_p)
    targetZ_p = targetZ.ctypes.data_as(c_double_p)
    targetW_p = targetW.ctypes.data_as(c_double_p)
    
    
    sourceX_p = sourceX.ctypes.data_as(c_double_p)
    sourceY_p = sourceY.ctypes.data_as(c_double_p)
    sourceZ_p = sourceZ.ctypes.data_as(c_double_p)
    sourceW_p = sourceW.ctypes.data_as(c_double_p)
    
    resultArray = np.zeros(numTargets)
    resultArray_p = resultArray.ctypes.data_as(c_double_p)

    _convolutionRoutines.directSum(ctypes.c_int(numTargets), ctypes.c_int(numSources),
                                                 targetX_p, targetY_p, targetZ_p, targetW_p,
                                                 sourceX_p, sourceY_p, sourceZ_p, sourceW_p,
                                                 resultArray_p)


    return resultArray


def callCompiledC_directSum_PoissonSingularitySubtract(numTargets, numSources, alphasq, targetX, targetY, targetZ, targetValue, targetWeight, sourceX, sourceY, sourceZ, sourceValue, sourceWeight):
    #                                                 (numTargets, numSources, alphasq, targetX, targetY, targetZ, targetW, sourceX, sourceY, sourceZ, sourceW)
   
    global _convolutionRoutines
    c_double_p = ctypes.POINTER(ctypes.c_double)

    
    targetX_p = targetX.ctypes.data_as(c_double_p)
    targetY_p = targetY.ctypes.data_as(c_double_p)
    targetZ_p = targetZ.ctypes.data_as(c_double_p)
    targetValue_p = targetValue.ctypes.data_as(c_double_p)
    targetWeight_p = targetWeight.ctypes.data_as(c_double_p)
    
    
    sourceX_p = sourceX.ctypes.data_as(c_double_p)
    sourceY_p = sourceY.ctypes.data_as(c_double_p)
    sourceZ_p = sourceZ.ctypes.data_as(c_double_p)
    sourceValue_p = sourceValue.ctypes.data_as(c_double_p)
    sourceWeight_p = sourceWeight.ctypes.data_as(c_double_p)
    
    resultArray = np.zeros(numTargets)
    resultArray_p = resultArray.ctypes.data_as(c_double_p)

    _convolutionRoutines.directSum_PoissonSingularitySubtraction(ctypes.c_int(numTargets), ctypes.c_int(numSources), ctypes.c_double(alphasq),
                                                 targetX_p, targetY_p, targetZ_p, targetValue_p, targetWeight_p,
                                                 sourceX_p, sourceY_p, sourceZ_p, sourceValue_p, sourceWeight_p,
                                                 resultArray_p)

    
    return resultArray


def callCompiledC_directSum_Poisson(numTargets, numSources, targetX, targetY, targetZ, targetValue, targetWeight, sourceX, sourceY, sourceZ, sourceValue, sourceWeight):
    #                                                 (numTargets, numSources, alphasq, targetX, targetY, targetZ, targetW, sourceX, sourceY, sourceZ, sourceW)
   
    global _convolutionRoutines
    c_double_p = ctypes.POINTER(ctypes.c_double)

    
    targetX_p = targetX.ctypes.data_as(c_double_p)
    targetY_p = targetY.ctypes.data_as(c_double_p)
    targetZ_p = targetZ.ctypes.data_as(c_double_p)
    targetValue_p = targetValue.ctypes.data_as(c_double_p)
    targetWeight_p = targetWeight.ctypes.data_as(c_double_p)
    
    
    sourceX_p = sourceX.ctypes.data_as(c_double_p)
    sourceY_p = sourceY.ctypes.data_as(c_double_p)
    sourceZ_p = sourceZ.ctypes.data_as(c_double_p)
    sourceValue_p = sourceValue.ctypes.data_as(c_double_p)
    sourceWeight_p = sourceWeight.ctypes.data_as(c_double_p)
    
    resultArray = np.zeros(numTargets)
    resultArray_p = resultArray.ctypes.data_as(c_double_p)

    _convolutionRoutines.directSum_Poisson(ctypes.c_int(numTargets), ctypes.c_int(numSources),
                                                 targetX_p, targetY_p, targetZ_p, targetValue_p, targetWeight_p,
                                                 sourceX_p, sourceY_p, sourceZ_p, sourceValue_p, sourceWeight_p,
                                                 resultArray_p)

    
    return resultArray

def callCompiledC_directSum_HelmholtzSingularitySubtract(numTargets, numSources, helmholtzK, targetX, targetY, targetZ, targetValue, targetWeight, sourceX, sourceY, sourceZ, sourceValue, sourceWeight):
    #                                                 (numTargets, numSources, alphasq, targetX, targetY, targetZ, targetW, sourceX, sourceY, sourceZ, sourceW)
   
    global _convolutionRoutines
    c_double_p = ctypes.POINTER(ctypes.c_double)
    
    
    targetX_p = targetX.ctypes.data_as(c_double_p)
    targetY_p = targetY.ctypes.data_as(c_double_p)
    targetZ_p = targetZ.ctypes.data_as(c_double_p)
    targetValue_p = targetValue.ctypes.data_as(c_double_p)
    targetWeight_p = targetWeight.ctypes.data_as(c_double_p)
    
    
    sourceX_p = sourceX.ctypes.data_as(c_double_p)
    sourceY_p = sourceY.ctypes.data_as(c_double_p)
    sourceZ_p = sourceZ.ctypes.data_as(c_double_p)
    sourceValue_p = sourceValue.ctypes.data_as(c_double_p)
    sourceWeight_p = sourceWeight.ctypes.data_as(c_double_p)
    
    resultArray = np.zeros(numTargets)
    resultArray_p = resultArray.ctypes.data_as(c_double_p)

    _convolutionRoutines.directSum_HelmholtzSingularitySubtraction(ctypes.c_int(numTargets), ctypes.c_int(numSources), ctypes.c_double(helmholtzK),
                                                 targetX_p, targetY_p, targetZ_p, targetValue_p, targetWeight_p,
                                                 sourceX_p, sourceY_p, sourceZ_p, sourceValue_p, sourceWeight_p,
                                                 resultArray_p)
 
     
    return resultArray




def callCompiledC_directSum_Helmholtz(numTargets, numSources, helmholtzK, targetX, targetY, targetZ, targetValue, targetWeight, sourceX, sourceY, sourceZ, sourceValue, sourceWeight):
    #                                                 (numTargets, numSources, alphasq, targetX, targetY, targetZ, targetW, sourceX, sourceY, sourceZ, sourceW)
   
    global _convolutionRoutines
    c_double_p = ctypes.POINTER(ctypes.c_double)
    
    
    targetX_p = targetX.ctypes.data_as(c_double_p)
    targetY_p = targetY.ctypes.data_as(c_double_p)
    targetZ_p = targetZ.ctypes.data_as(c_double_p)
    targetValue_p = targetValue.ctypes.data_as(c_double_p)
    targetWeight_p = targetWeight.ctypes.data_as(c_double_p)
    
    
    sourceX_p = sourceX.ctypes.data_as(c_double_p)
    sourceY_p = sourceY.ctypes.data_as(c_double_p)
    sourceZ_p = sourceZ.ctypes.data_as(c_double_p)
    sourceValue_p = sourceValue.ctypes.data_as(c_double_p)
    sourceWeight_p = sourceWeight.ctypes.data_as(c_double_p)
    
    resultArray = np.zeros(numTargets)
    resultArray_p = resultArray.ctypes.data_as(c_double_p)

    _convolutionRoutines.directSum_Helmholtz(ctypes.c_int(numTargets), ctypes.c_int(numSources), ctypes.c_double(helmholtzK),
                                                 targetX_p, targetY_p, targetZ_p, targetValue_p, targetWeight_p,
                                                 sourceX_p, sourceY_p, sourceZ_p, sourceValue_p, sourceWeight_p,
                                                 resultArray_p)
 
     
    return resultArray



if __name__=="__main__":
    import os
    myhost = os.uname()[1]
    print(myhost)
    cwd = os.getcwd()
    print(cwd)
    
    numTargets = numSources = 2000
    x = np.random.rand(numTargets)
    y = np.random.rand(numTargets)
    z = np.random.rand(numTargets)
    w = np.ones(numTargets)
    
    alphasq = 1.0
    
    start = time.time()
#     result = treeEval_Python(numTargets, numSources, x,y,z,w, x,y,z,w)
    result = callCompiledC_directSum_PoissonSingularitySubtract(numTargets, numSources, alphasq, x,y,z,w, x,y,z,w)
    includingWrapperTime = time.time()-start
    print('Time spent in wrapper function: ', includingWrapperTime)
    
    print('C routine result:')
    print(result[:5])
    
    # perform direct sum here in python and compare
    pythonResultsArray = np.zeros(numTargets)
    startPython = time.time()
    for i in range(numTargets):
        for j in range(numSources):
            r = np.sqrt( (x[i]-x[j])**2 + (y[i]-y[j])**2 + (z[i]-z[j])**2 )
            if r > 0.0:
                pythonResultsArray[i] += w[j]/r
    pythonTime = time.time()-startPython
    print('Python time: ', pythonTime)
      
    print('Python routine result:')
    print(pythonResultsArray[:5])
      
    print('Max Difference: ', np.max(result-pythonResultsArray) )
