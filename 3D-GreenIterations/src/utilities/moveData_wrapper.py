import numpy as np
import ctypes
from mpi4py import MPI

from mpiUtilities import rprint


try: 
    moveDataRoutines = ctypes.CDLL('libMoveData.so')
except OSError as e:
    print(e)
    exit(-1)
    
c_double_p  = ctypes.POINTER(ctypes.c_double) 
c_double_pp = np.ctypeslib.ndpointer(dtype=np.intp)
        

        
""" Set argtypes of the wrappers. """

 
try:
    moveDataRoutines.copyToDevice.argtypes = ( 
                                              ctypes.POINTER(ctypes.c_double), 
                                              ctypes.POINTER(ctypes.c_double), 
                                              ctypes.POINTER(ctypes.c_double), 
                                              ctypes.POINTER(ctypes.c_double), 
                                              np.ctypeslib.ndpointer(dtype=np.intp),
                                              ctypes.c_int, 
                                              ctypes.c_int
                                              ) 
    
    moveDataRoutines.copyFromDevice.argtypes = ( 
                                              ctypes.POINTER(ctypes.c_double), 
                                              ctypes.POINTER(ctypes.c_double), 
                                              ctypes.POINTER(ctypes.c_double), 
                                              ctypes.POINTER(ctypes.c_double), 
                                              np.ctypeslib.ndpointer(dtype=np.intp),
                                              ctypes.c_int, 
                                              ctypes.c_int
                                              ) 
    
    moveDataRoutines.removeVectorFromDevice.argtypes = (
                                                        ctypes.POINTER(ctypes.c_double),
                                                        ctypes.c_int
                                                        )
    
    moveDataRoutines.removeMatrixFromDevice.argtypes = (
                                                        np.ctypeslib.ndpointer(dtype=np.intp),
                                                        ctypes.c_int,
                                                        ctypes.c_int
                                                        )
    
    moveDataRoutines.copyVectorToDevice.argtypes = (
                                                        ctypes.POINTER(ctypes.c_double),
                                                        ctypes.c_int
                                                        )
    
    moveDataRoutines.copyMatrixToDevice.argtypes = (
                                                        np.ctypeslib.ndpointer(dtype=np.intp),
                                                        ctypes.c_int,
                                                        ctypes.c_int
                                                        )
    
    moveDataRoutines.copyVectorFromDevice.argtypes = (
                                                        ctypes.POINTER(ctypes.c_double),
                                                        ctypes.c_int
                                                        )
    
    moveDataRoutines.copyMatrixFromDevice.argtypes = (
                                                        np.ctypeslib.ndpointer(dtype=np.intp),
                                                        ctypes.c_int,
                                                        ctypes.c_int
                                                        )
    
except NameError:
    print("Could not set argtypes of moveDataRoutines")




def callCopyToDevice(X,Y,Z,W,wavefunctions):

    numWavefunctions,numPoints = np.shape(wavefunctions)
    
    X_p = X.ctypes.data_as(c_double_p)
    Y_p = X.ctypes.data_as(c_double_p)
    Z_p = X.ctypes.data_as(c_double_p)
    W_p = X.ctypes.data_as(c_double_p)
    wavefunctions_pp = (wavefunctions.__array_interface__['data'][0] + np.arange(wavefunctions.shape[0])*wavefunctions.strides[0]).astype(np.intp)
    
    moveDataRoutines.copyToDevice(X_p, Y_p, Z_p, W_p, wavefunctions_pp, numPoints, numWavefunctions )
    
    return 


def callCopyFromDevice(X,Y,Z,W,wavefunctions):

    numWavefunctions,numPoints = np.shape(wavefunctions)
    
    X_p = X.ctypes.data_as(c_double_p)
    Y_p = X.ctypes.data_as(c_double_p)
    Z_p = X.ctypes.data_as(c_double_p)
    W_p = X.ctypes.data_as(c_double_p)
    wavefunctions_pp = (wavefunctions.__array_interface__['data'][0] + np.arange(wavefunctions.shape[0])*wavefunctions.strides[0]).astype(np.intp)
    
    moveDataRoutines.copyFromDevice(X_p, Y_p, Z_p, W_p, wavefunctions_pp, numPoints, numWavefunctions )
    
    return X,Y,Z,W,wavefunctions


def callRemoveVectorFromDevice(U):

    numPoints = len(U)
    U_p = U.ctypes.data_as(c_double_p)
    moveDataRoutines.removeVectorFromDevice(U_p, numPoints)
    
    return 

def callRemoveMatrixFromDevice(wavefunctions):

    numWavefunctions,numPoints = np.shape(wavefunctions)
    wavefunctions_pp = (wavefunctions.__array_interface__['data'][0] + np.arange(wavefunctions.shape[0])*wavefunctions.strides[0]).astype(np.intp)
    moveDataRoutines.removeMatrixFromDevice(wavefunctions_pp, numPoints, numWavefunctions)
    
    return 

def callCopyVectorToDevice(U):

    numPoints = len(U)
    U_p = U.ctypes.data_as(c_double_p)
    moveDataRoutines.copyVectorToDevice(U_p, numPoints)
    
    return 

def callCopyVectorFromDevice(U):

    numPoints = len(U)
    U_p = U.ctypes.data_as(c_double_p)
    moveDataRoutines.copyVectorFromDevice(U_p, numPoints)
    
    return 

def callCopyMatrixToDevice(wavefunctions):

    numWavefunctions,numPoints = np.shape(wavefunctions)
    wavefunctions_pp = (wavefunctions.__array_interface__['data'][0] + np.arange(wavefunctions.shape[0])*wavefunctions.strides[0]).astype(np.intp)
    moveDataRoutines.copyMatrixToDevice(wavefunctions_pp, numPoints, numWavefunctions)
    print("\n\n Inside callCopyMatrixToDevice, wavefunctions_pp = ",wavefunctions_pp,"\n\n")

    return 

def callCopyMatrixFromDevice(wavefunctions):

    numWavefunctions,numPoints = np.shape(wavefunctions)
    wavefunctions_pp = (wavefunctions.__array_interface__['data'][0] + np.arange(wavefunctions.shape[0])*wavefunctions.strides[0]).astype(np.intp)
    moveDataRoutines.copyMatrixFromDevice(wavefunctions_pp, numPoints, numWavefunctions)
    
    return 










