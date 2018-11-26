'''
Density mixing scheme

@author: nathanvaughn
'''
from numpy import pi, cos, arccos, sin, sqrt, exp
import numpy as np
from scipy.special import factorial, comb

import matplotlib.pyplot as plt


def innerProduct(f,g,weights):
    return np.sum( f*g*weights )

def computeFarray(inputDensities, outputDensities):
    F = np.zeros_like(inputDensities)
    (M,n) = np.shape(F)
    
    for i in range(n):
        F[:,i] = outputDensities[:,i] - inputDensities[:,i]
    return F


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
        rhs[m] = innerProduct(f, F[:,n-1], weights)
        for k in range(n-1): # k columns
            g = F[:,n-1] - F[:,n-2-k]
            linearSystem[k,m] = innerProduct( f, g, weights)
    
#     print('\nLinear system: ', linearSystem)
#     print('\nrhs: ', rhs)
    cvec = np.linalg.solve(linearSystem, rhs)
    print('\nsolution: ', cvec)
    return cvec
    


def computeNewDensity(inputDensities, outputDensities, mixingParameter,weights):
    '''
    :param cvec: vector of weights, resulting from the minimization
    :param inputDensity: array of input densities.  After nth SCF, it has shape (M,n) where M is the number of quadrature points
    :param outputDensity: similar to inputDensities
    
    :returns input density for the (n+1) SCF iteration
    '''
    
    (M,n) = np.shape(inputDensities)
    print('Input densities has shape ', np.shape(inputDensities))
    
    F = computeFarray(inputDensities, outputDensities)
    cvec = solveLinearSystem(F,weights)
    
    weightedInputDensity  = np.copy(inputDensities[:,n-1])
    weightedOutputDensity = np.copy(outputDensities[:,n-1])
    
    for k in range(0,n-1):
#         print('\nk = ',k)
#         print(cvec[k])
#         print(inputDensities[:,n-2-k])
#         print(inputDensities[:,n-1])
#         print(outputDensities[:,n-2-k])
#         print(outputDensities[:,n-1])
#         print()
        weightedInputDensity += cvec[k] * (inputDensities[:,n-2-k] - inputDensities[:,n-1]) 
        weightedOutputDensity += cvec[k] * (outputDensities[:,n-2-k] - outputDensities[:,n-1]) 
    
    
    return mixingParameter*weightedOutputDensity + (1-mixingParameter)*weightedInputDensity
 
if __name__=="__main__":
    mixingParameter = 0.5
    M = 100
    n = 3
    xvec = np.linspace(0,1,M)
    weights = (1/M)*np.ones(M)
    inputDensities = np.zeros((M,n))
    outputDensities = np.zeros((M,n))
    
    for i in range(n):
        if i==0:
            inputDensities[:,i] = 1-xvec**(i+1)
            outputDensities[:,i] = 0-xvec**(5*(i+1))
        else:   
            inputDensities[:,i] = xvec**(i+1)
            outputDensities[:,i] = xvec**(1.01*(i+1))
        
#         print(inputDensities[:,i])
#         print(outputDensities[:,i])
        
        plt.plot(xvec,inputDensities[:,i],label='input   %i' %i)
        plt.plot(xvec,outputDensities[:,i],'-.',label='output %i' %i)
        

    
    
    target = n-1
#     outputDensities[:,target] = inputDensities[:,target] - 0.00
#     outputDensities[:,target+1] = inputDensities[:,target+1]+ 1.01
    nextDensity = computeNewDensity(inputDensities, outputDensities, mixingParameter,weights)
#     print('\nNext density: \n', nextDensity)
#     print('\nThe input=output density: \n', inputDensities[:,target])
#     print('\nThe input=output density: \n', inputDensities[:,target+1])
    
#     plt.figure()
#     plt.plot(xvec,outputDensities[:,target],'b')
#     plt.plot(xvec,inputDensities[:,target],'b-.')
#     plt.plot(xvec,outputDensities[:,target+1],'r')
#     plt.plot(xvec,inputDensities[:,target+1],'r-.')
    plt.plot(xvec,nextDensity,'k')
    plt.legend()
    plt.show()
    
    plt.close()

    
    