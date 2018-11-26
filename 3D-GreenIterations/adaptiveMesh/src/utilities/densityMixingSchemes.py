'''
Density mixing scheme

@author: nathanvaughn
'''
from numpy import pi, cos, arccos, sin, sqrt, exp
import numpy as np
from scipy.special import factorial, comb


def innerProduct(f,g,weights):
    return np.sum( f*g*weights )


def solveLinearSystem(F,weights):
    '''
    
    :param F: array of F's, where the nth column is rho_out - rho_in from the nth SCF iteration
    :param weights:
    '''
    (M,n) = np.shape(F)
    print('F has shape ', np.shape(inputDensities))
    
    linearSystem = np.zeros((n-1,n-1))
    rhs = np.zeros(n-1)
    for m in range(n-1): # m rows
        f = F[:,n-1] - F[:,n-2-m]
        rhs[m] = innerproduct(f, F[:,n-1])
        for k in range(n-1): # k columns
            g = F[:,n-1] - F[:,n-2-k]
            linearSystem[k,m] = innerProduct( f, g, weights)
    
    print('Linear system: ', linearSystem)
    print('rhs: ', rhs)
    cvec = np.linalg.solve(linearSystem, rhs)
    print('solution: ', cvec)
    return cvec
    


def computeNewDensity(cvec, inputDensities, outputDensities, mixingParameter):
    '''
    :param cvec: vector of weights, resulting from the minimization
    :param inputDensity: array of input densities.  After nth SCF, it has shape (M,n) where M is the number of quadrature points
    :param outputDensity: similar to inputDensities
    
    :returns input density for the (n+1) SCF iteration
    '''
    
    (M,n) = np.shape(inputDensities)
    print('Input densities has shape ', np.shape(inputDensities))
    
    weightedInputDensity  = inputDensities[:,n-1]
    weightedOutputDensity = outputDensities[:,n-1]
    
    for k in range(0,n-1):
        weightedInputDensity += cvec[k] * (inputDensities[:,n-2-k] - inputDensities[:,n-1]) 
        weightedOutputDensity += cvec[k] * (outputDensities[:,n-2-k] - outputDensities[:,n-1]) 
    
    
    return mixingParameter*weightedOutputDensity + (1-mixingParameter)*weightedInputDensity
 
if __name__=="__main__":
    return

    
    