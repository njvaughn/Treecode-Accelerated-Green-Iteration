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
        F[:,i] =  outputDensities[:,i] - inputDensities[:,i] 

    return F


def solveLinearSystem(F,weights):
    '''
    
    :param F: array of F's, where the nth column is rho_out - rho_in from the nth SCF iteration
    :param weights:
    '''
    (M,n) = np.shape(F)
#     print('F has shape ', np.shape(F))
    
    linearSystem = np.zeros((n-1,n-1))
    rhs = np.zeros(n-1)
    for m in range(n-1): # m rows
        f = F[:,n-1] - F[:,n-2-m]
        rhs[m] = innerProduct(f, F[:,n-1], weights)
        for k in range(n-1): # k columns
            g = F[:,n-1] - F[:,n-2-k]
            linearSystem[m,k] = innerProduct( f, g, weights)
    
#     print('\nLinear system: ', linearSystem)
#     print('\nrhs: ', rhs)
    cvec = np.linalg.solve(linearSystem, rhs)
    print('\nAnderson weights: ', cvec[::-1] , 1-np.sum(cvec))
    return cvec
    


def computeNewDensity(inputDensities, outputDensities, mixingParameter,weights, returnWeights=False):
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
        
    nextDensity = mixingParameter*weightedOutputDensity + (1-mixingParameter)*weightedInputDensity
    if returnWeights==True:
        return nextDensity, cvec
    elif returnWeights==False:
        return nextDensity
    
def AitkenAcceleration(a, b, c):
    
    
#     return new - (middle - old)**2 / (new - 2*middle + old)
    return a - (b - a)**2 / (a - 2*b + c)
 

def test1():
    mixingParameter = 0.5
    M = 100
    n = 4
    xvec = np.linspace(0,1,M)
    weights = (1/M)*np.ones(M)
#     inputDensities = np.zeros((M,n))
#     outputDensities = np.zeros((M,n))
    
    inputDensities = np.zeros((M,1))
    outputDensities = np.zeros((M,1))
    
    print(np.shape(inputDensities))
    
    for i in range(n):
        if i==0:
            inputDensities[:,i] = xvec**(i+1) 
            outputDensities[:,i] =  1.1*xvec**((i+1))
        else:   
#             print('shape of xvec**(i+1): ', np.shape(xvec**(i+1)))
            v = np.reshape(xvec**(i+1), (M,1))
            vv = np.reshape(1.1*xvec**(1*(i+1)), (M,1))
            inputDensities= np.concatenate((inputDensities, v), axis=1)
            outputDensities = np.concatenate((outputDensities,vv), axis=1)
            
#         print(np.shape(inputDensities))
        
#         print(inputDensities[:,i])
#         print(outputDensities[:,i])

        nextDensity = computeNewDensity(inputDensities, outputDensities, mixingParameter,weights)
        plt.plot(xvec,nextDensity,label='input after %i' %i)
        
        plt.plot(xvec,inputDensities[:,i],label='input   %i' %i)
        plt.plot(xvec,outputDensities[:,i],'-.',label='output %i' %i)
        
        

    
    
#     target = n-1
# #     outputDensities[:,target] = inputDensities[:,target] - 0.00
# #     outputDensities[:,target+1] = inputDensities[:,target+1]+ 1.01
#     nextDensity = computeNewDensity(inputDensities, outputDensities, mixingParameter,weights)
# #     print('\nNext density: \n', nextDensity)
# #     print('\nThe input=output density: \n', inputDensities[:,target])
# #     print('\nThe input=output density: \n', inputDensities[:,target+1])
#     
# #     plt.figure()
# #     plt.plot(xvec,outputDensities[:,target],'b')
# #     plt.plot(xvec,inputDensities[:,target],'b-.')
# #     plt.plot(xvec,outputDensities[:,target+1],'r')
# #     plt.plot(xvec,inputDensities[:,target+1],'r-.')
#     plt.plot(xvec,nextDensity,'k')
    plt.legend()
    plt.show()
    
    
def test2():
    mixingParameter = 0.5
    M = 100
    n = 2
    xvec = np.linspace(0,1,M)
    weights = (1/M)*np.ones(M)
#     inputDensities = np.zeros((M,n))
#     outputDensities = np.zeros((M,n))
    
    inputDensities = np.zeros((M,1))
    outputDensities = np.zeros((M,1))
    
    print(np.shape(inputDensities))
    
    for i in range(n):
        if i==0:
            inputDensities[:,i] = xvec**(i+1) 
            outputDensities[:,i] =  0.9*xvec**((i+1))
        else:   
#             print('shape of xvec**(i+1): ', np.shape(xvec**(i+1)))
            v = np.reshape(xvec**(i+1), (M,1))
            vv = np.reshape(1.4*xvec**(1*(i+1)), (M,1))
            inputDensities= np.concatenate((inputDensities, v), axis=1)
            outputDensities = np.concatenate((outputDensities,vv), axis=1)
            
#         print(np.shape(inputDensities))
        
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


if __name__=="__main__":
    
    test2()
    

    
    