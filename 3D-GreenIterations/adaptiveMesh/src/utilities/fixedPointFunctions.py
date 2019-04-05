'''
Created on Jan 16, 2019

@author: nathanvaughn
'''
import numpy as np
from scipy.optimize import anderson


def PsiNorm(psi):
    return np.sqrt( np.sum(psi*psi*weights) )

   
def PowerIteration(psiIn):
    psiOut = np.dot(A,psiIn)
    psiOut /= np.linalg.norm(psiOut)
    return psiOut - psiIn

def PowerIteration2(psiIn):
#     print(psi1)
#     print(weights)
    psiOut = np.dot(A,psiIn)
    psiOut /= np.sqrt( np.dot(psiOut,psiOut) )
    psiOut -= ( np.dot(psiOut, psi1) )/( np.dot(psi1, psi1) ) * psi1
#     psiOut -= np.sqrt( np.dot(psiOut, psi1) )/np.sqrt( np.dot(psi1, psi1) ) * psi1
#     dot = np.dot(psiOut, psi1)
#     print('dot ', dot)
#     overlap=np.sqrt( np.dot(psiOut, psi1) )/ np.linalg.norm(psi1) 
#     print(overlap)
#     psiOut -= overlap * psi1
#     psiOut /= np.linalg.norm(psiOut)
    psiOut /= np.sqrt( np.dot(psiOut,psiOut) )
    
#     r = np.dot( psiOut, np.dot(A,psiOut)) 
#     
#     psiOut *= np.sign(r)
#     print(psiOut)
#     print()

    if np.linalg.norm(psiOut - psiIn) < np.linalg.norm(psiOut + psiIn):
        return psiOut - psiIn
    else:
        psiOut *= -1
        return psiOut - psiIn


def test(N):
#     psiIn = np.random.rand(N)
    global A, psi1, weights
    weights = np.ones(N)
    A = np.random.rand(N,N)
    A = A+A.T
    
    ## First Test Power Iterations
    psi1 = np.random.rand(N)
    psi1 /= np.linalg.norm(psi1)
    residual=1
    count=1
    while ( (residual > 1e-7) and (count<10000) ):
        r = PowerIteration(psi1)
        psi1 += r
        residual = np.linalg.norm(r)
        print('Iteration %i: Residual = %1.2e' %(count,residual))
        count+=1
    eig1=np.dot( psi1, np.dot(A,psi1)) / np.dot(psi1,psi1)
    print('Rayleigh Quotient: ', eig1)
#     psi1 = np.copy(psiIn)
    print()
    print()
    psi2 = np.random.rand(N)
    psi2 /= np.linalg.norm(psi2)
    print(psi2)
    residual=1
    count=1
    while ( (residual > 1e-7) and (count<10000) ):
        r = PowerIteration2(psi2)
        psi2 += r
        residual = np.linalg.norm(r)
        print('Iteration %i: Residual = %1.2e' %(count,residual))
        count+=1
    eig2=np.dot( psi2, np.dot(A,psi2)) / np.dot(psi2,psi2)
    print('Rayleigh Quotient: ', eig2)
    
#     print('psiIn: ', psiIn)
    print()
    psiIn = np.random.rand(N)
    psiIn /= np.linalg.norm(psiIn)
    psi1 = anderson(PowerIteration,psiIn,M=10,w0=0.01,tol_norm=np.linalg.norm,f_tol=1e-7,verbose=True)
    print('Rayleigh Quotient: ', np.dot( psi1, np.dot(A,psi1)) / np.dot(psi1,psi1))
    print('Difference: ', np.dot( psi1, np.dot(A,psi1)) / np.dot(psi1,psi1)-eig1)
    print()
    print()
     
    psiIn = np.random.rand(N)
    psiIn /= np.linalg.norm(psiIn)
    
    residual=1
    count=1
    while ( (residual > 1e-1) and (count<10000) ):
        r = PowerIteration2(psiIn)
        psiIn += r
        residual = np.linalg.norm(r)
#         print('Iteration %i: Residual = %1.2e' %(count,residual))
        count+=1
#     print('Rayleigh Quotient: ', np.dot( psi2, np.dot(A,psi2)) / np.dot(psi2,psi2))
    
    
    psi2 = anderson(PowerIteration2,psiIn,M=10, w0=0.01,tol_norm=np.linalg.norm,f_tol=1e-7,verbose=True)
    print('Rayleigh Quotient: ', np.dot( psi2, np.dot(A,psi2)) / np.dot(psi2,psi2))
    print('Difference: ', np.dot( psi2, np.dot(A,psi2)) / np.dot(psi2,psi2)-eig2)
    print('Used %i iterations in preprocessing.' %count)
#     print('psi1 = ', psi1)
#     print('psi2 = ', psi2)
#     print('Overlap: ', np.dot(psi1,psi2))



def GreenIteration(m,orbitals,):
    return

if __name__=="__main__":
    
    test(10)