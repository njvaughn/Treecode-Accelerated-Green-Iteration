'''
Density mixing scheme

@author: nathanvaughn
'''
from numpy import pi, cos, arccos, sin, sqrt, exp
import numpy as np
from scipy.special import factorial, comb

import matplotlib.pyplot as plt
from mpmath.calculus.optimization import steffensen


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
    numerator = (b - a)**2
    denominator = (a - 2*b + c)
#     print(np.shape(numerator))
#     print(np.shape(denominator))
#     if abs(denominator)<1e-16: 
    if abs(denominator).any()<1e-13: 
        print('Warning, abs(denominator) < 1e-15')
        return a
    correction = numerator / denominator
    
    print(np.max(correction))
    print(np.min(correction))
    
#     if abs(numerator).all()<1e-16: print('Warning, abs(numerator) < 1e-16')
#     if abs(denominator).all()<1e-16: print('Warning, abs(denominator) < 1e-16')
#     if abs(numerator)<1e-16: print('Warning, abs(numerator) < 1e-16')
    
#     print('Correction: ', correction)
    return a - correction

#     return (a*c-b*b) / ( c - 2*b + a )
 

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


def testSteffensenScalar():
    
    def f(x):
#         return (np.array(x) + 2/np.array(x)) / 2
        return 6.28 + np.sin(x)
    
    xold=-100
    count=1
    residual=1
    residualVec=[]
    while residual>1e-14:
        xnew = f(xold)
        residual = abs(xnew-xold)
        residualVec.append(residual)
        xold = np.copy(xnew)
        print('Iteration %2i, x = %1.10f, residual = %1.3e' %(count, xnew, residual))
        count+=1
    plt.semilogy(residualVec, label='Original')
    print()
    print()
    xold=-100
    count=1
    residual=1
    residualVec=[]
    while residual>1e-14:
        fx = f(xold)
        fxx = f(fx)
        xnew = AitkenAcceleration(xold, fx, fxx)
        residual = abs(xnew-xold)
        residualVec.append(residual)
        xold = np.copy(xnew)
        print('Iteration %2i, x = %1.10f, residual = %1.3e' %(count, xnew, residual))
        count+=1
    plt.semilogy(residualVec, label='Steffensen')
    plt.legend()
    plt.title('Convergence of Fixed Point Iteration: x = 6.28 + sin(x)')
    plt.xlabel('Iteration Count')
    plt.ylabel('Residual')
    plt.show()
        
def testSteffenson(N):
    if N==2:
        A = np.zeros((2,2))
        A[0,0]=1
        A[1,1]=0.9
        A[1,0]=2
        A[0,1]=0
    else:
        A = np.random.rand(N,N)
#         A = (A + A.T)/2
    x = np.random.rand(N)
    x /= np.linalg.norm(x)
    xs = np.copy(x)
    
    eigs = np.linalg.eigvals(A)
#     print(eigs)
    
    ## preprocess to get an accurate eigenvalue and eigenvector
    vectorResidual=1
    eigOld=1
    t = np.random.rand(N)
    count=1
    limit=1000

    while ( (vectorResidual>1e-12) and (count<limit) ):
        y = np.dot(A,t)
        y /= np.linalg.norm(y)
        eig = np.dot(y, np.dot(A,y))
        residual = abs( eig-eigOld )
        eigOld=eig
        vectorResidual = np.linalg.norm(t-y)
#         print(count, ': ', eig, ', residual: ', residual)
#         print('Power Iteration %2i, Eigenvalue: %1.10f, Eigenvector residual: %1.3e, Eigenvalue residual: %1.3e' %(count,eig,vectorResidual,residual))
        t = np.copy(y)
        count+=1
    e = eig
    print('Converged Eigenfunction and eigenvalue saves as (t,e). e=', e)
    print()
    print()
#     e = 1
    
        
    errorVec = []
    residualVec = []
    count=1
    residual=1
    eigOld = 100
    vectorResidual=1
    while ( (vectorResidual>1e-12) and (count<limit) ):
        y = np.dot(A,x)
        y /= np.linalg.norm(y)
        eig = np.dot(y, np.dot(A,y))
        residual = abs( eig-eigOld )
        eigOld=eig
        vectorResidual = np.linalg.norm(x-y)
        errorNorm = np.linalg.norm(y-t)
        errorVec.append(eig-e)
        residualVec.append(vectorResidual)
#         print('Power Iteration %2i, Eigenvalue: %1.10f, Eigenvector residual: %1.3e, Eigenvalue residual: %1.3e' %(count,eig,vectorResidual,residual))
#         print('Power Iteration %2i, Eigenvalue Error: %1.12f, Eigenvector Error: %1.12f' %(count,abs(eig-e),errorNorm))
        x = np.copy(y)
        count+=1
    print('Power iteration eig = ', eig)
    plt.semilogy(residualVec,label="Power Iteration")
    powerIterationCount = count
    
#     print() 
#     print('Error vec: ')
#     print(np.array(errorVec))
#     ratioVec = np.zeros(len(errorVec)-1)
#     for i in range(len(errorVec)-1):
#         ratioVec[i] = errorVec[i]/errorVec[i+1]
#     print()
#     print('Ratio Vec: ')
#     print(ratioVec)
#     print() 
#     print()

    print() 
    print('Residual vec: ')
    print(np.array(residualVec))
    ratioVec = np.zeros(len(residualVec)-1)
    for i in range(len(residualVec)-1):
        ratioVec[i] = residualVec[i]/residualVec[i+1]
    print()
    print('Ratio Vec: ')
    print(ratioVec)
    print() 
    print()
    
    
    residualVec = []
    errorVec = []
    x = np.copy(xs)  
    count=1
    residual=1
    eigOld = 100
    vectorResidual=1
    while ( (vectorResidual>1e-12) and (count<limit) ):
        xold = np.copy(x)
        
        y = np.dot(A,x)
        y /= np.linalg.norm(y)
#         print('y eig: %1.10f' %(np.dot(y, np.dot(A,y))))py
        z = np.dot(A,y)
        z /= np.linalg.norm(z)
#         print('z eig: %1.10f' %(np.dot(z, np.dot(A,z))))
        
        x = AitkenAcceleration(xold,y,z)
        x /= np.linalg.norm(x)
        
        # Throw in an extra iteration
        x = np.dot(A,x)
        x /= np.linalg.norm(x)

#         print('Norm of aitken x: ', np.linalg.norm(x))
        eig = np.dot(x, np.dot(A,x))
        residual = abs( eig-eigOld )
        eigOld=eig
        vectorResidual = np.linalg.norm(x-xold)
        residualVec.append(vectorResidual)
        errorNorm = np.linalg.norm(x-t)
        errorVec.append(eig-e)
#         print('Steffensen Iteration %2i, Eigenvalue: %1.10f, Eigenvector residual: %1.3e, Eigenvalue residual: %1.3e' %(count,eig,vectorResidual,residual))
#         print('Steffensen Iteration %2i, Eigenvalue Error: %1.12f, Eigenvector Error: %1.12f' %(count,abs(eig-e),errorNorm))
        count+=1
    print('Steffensen eig = ', eig)
    plt.semilogy(residualVec,'o',label="Steffensen Accelerated")
    plt.title('Power Iteration Convergence for Matrix with lamba1=%1.5f, lambda2=%1.5f' %(eigs[0],eigs[1]))
    plt.xlabel('Iteration Count')
    plt.ylabel('Eigenvector Residual Norm')
    plt.legend()
    
    steffensenCount = count
    
#     print() 
#     print(np.array(errorVec))
#     ratioVec = np.zeros(len(errorVec)-1)
#     for i in range(len(errorVec)-1):
#         ratioVec[i] = errorVec[i]/errorVec[i+1]
#     print()
#     print(ratioVec)
#     print()
    
    print() 
    print('Residual vec: ')
    print(np.array(residualVec))
    ratioVec = np.zeros(len(residualVec)-1)
    for i in range(len(residualVec)-1):
        ratioVec[i] = residualVec[i]/residualVec[i+1]
    print()
    print('Ratio Vec: ')
    print(ratioVec)
    print() 
    print() 
    
    print('True eigenvalues: ', eigs)
    print()
    print('Power iterations:                                        ', powerIterationCount)
    print('Steffensen iterations (times 2, since 2 matvecs per):    ', steffensenCount*2)
    plt.show()
if __name__=="__main__":
    
#     test2()
    testSteffenson(10)
#     testSteffensenScalar()
    

    
    