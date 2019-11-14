'''
Created on Jan 16, 2019

@author: nathanvaughn
'''
import numpy as np
from scipy.optimize import anderson as scipyAnderson
from scipy.optimize import root as scipyRoot
from scipy.optimize import show_options
from scipy.optimize.nonlin import BroydenFirst, KrylovJacobian
from scipy.optimize.nonlin import InverseJacobian

# global tree, orbitals, oldOrbitals

def PsiNorm(psi):
    return np.sqrt( np.sum(psi*psi*weights) )

global counter
counter=0   
def PowerIteration(psiIn,dummy,dummy2):
    print('Dummy = ', dummy)
    print('Dummy2 = ', dummy2)
    global counter
    counter += 1
    print(counter)
    psiOut = np.dot(A,psiIn)
    psiOut /= np.linalg.norm(psiOut)
    psiOut /= np.max(np.abs(psiOut))
    return psiOut - psiIn, dummy

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


def PowerIterationOptional(psiIn):
#     print('From inside PowerIterationOptional:')
#     print('psiOrth: ', psiOrth)
#     print('weights: ', weights)
    psiOut = np.dot(A,psiIn)
    psiOut /= np.sqrt( np.dot(psiOut,psiOut) )
    psiOut -= ( np.dot(psiOut, psiOrth) )/( np.dot(psiOrth, psiOrth) ) * psiOrth

#     psiOut /= np.sqrt( np.dot(psiOut,psiOut) )
    psiOut /= np.max(np.abs(psiOut))
    

    if np.linalg.norm(psiOut - psiIn) < np.linalg.norm(psiOut + psiIn):
        return psiOut - psiIn
    else:
        psiOut *= -1
        return psiOut - psiIn
    

# def PowerIteration_root(psiIn):
#     psiOut = np.dot(A,psiIn)
#     psiOut /= np.max(np.abs(psiOut))
#     return psiOut
    
def testRootfinders(N):
    global A, psi1, weights
    if N==3:
        A = np.array( [ [1,0,0],
                        [0,0.5,0],
                        [0,0,0.99]
                    ] )
    else:
        weights = np.ones(N)
        A = np.random.rand(N,N)
        A = A+A.T
    
    
    
    psiIn = np.random.rand(N)
    
    psiPower = np.copy(psiIn)
    psiPower /= np.max(np.abs(psiPower))
    residual=1
    count=1
    while ( (residual > 1e-12) ):
        r = PowerIteration(psiPower)
        psiPower += r
        residual = np.max(np.abs(r))
#         print('Iteration %i: Residual = %1.2e' %(count,residual))
        count+=1
    print('Power Iteration used %i iterations, residual = %1.2e\n\n' %(count,residual))

#     psiOutAnderson = scipyAnderson(PowerIteration, psiIn,f_tol=1e-12,verbose=True)

    
#     show_options(solver='root', method='anderson', disp=True)
    options={'fatol':1e-12}
#     options={'verbose':True}
    for method in ["anderson", "broyden1", "broyden2","hybr","lm","krylov"]:
        psiOutAndersonRoot = scipyRoot(PowerIteration, psiIn, method=method, options=options)
#         print(psiOutAndersonRoot.__dir__())
#         print(psiOutAndersonRoot.success)
#         print(psiOutAndersonRoot.message)
        try:
            print(method + ' number of evals: ', psiOutAndersonRoot.nit)
        except Exception:
            print(method + ' number of evals: ', psiOutAndersonRoot.nfev)
        
        
        diff1 = psiOutAndersonRoot.x-psiPower
        diff2 = psiOutAndersonRoot.x+psiPower
        norm1 = np.max(np.abs(diff1))
        norm2 = np.max(np.abs(diff2))
        norm = min(norm1,norm2)
        
        if norm>1e-8:
            print('Trying again...')
            psiIn = np.random.rand(N)
            psiIn -= np.dot(psiOutAndersonRoot.x,psiIn)/np.dot(psiIn,psiIn)
            psiIn /= np.max(np.abs(psiIn))
            psiOutAndersonRoot = scipyRoot(PowerIteration, psiIn, method=method, options=options)
            diff1 = psiOutAndersonRoot.x-psiPower
            diff2 = psiOutAndersonRoot.x+psiPower
            norm1 = np.max(np.abs(diff1))
            norm2 = np.max(np.abs(diff2))
            norm = min(norm1,norm2)
            try:
                print(method + ' number of evals: ', psiOutAndersonRoot.nit)
            except Exception:
                print(method + ' number of evals: ', psiOutAndersonRoot.nfev)
        print('Norm of difference: ', norm)
        print()
        
        
def testAndersonOptions(N):
    global A, psi1, weights
    if N==3:
        A = np.array( [ [1,0,0],
                        [0,0.5,0],
                        [0,0,0.99]
                    ] )
        psiIn=np.array([1,2,3])
    else:
        weights = np.ones(N)
        A = np.random.rand(N,N)
        A = A+A.T
    
    
    
        psiIn = np.random.rand(N)
    
    
#     psiPower = np.copy(psiIn)
#     psiPower /= np.max(np.abs(psiPower))
#     residual=1
#     count=1
#     while ( (residual > 1e-3) ):
#         r = PowerIteration(psiPower)
#         psiPower += r
#         residual = np.max(np.abs(r))
# #         print('Iteration %i: Residual = %1.2e' %(count,residual))
#         count+=1
#     print('Power Iteration used %i iterations, residual = %1.2e\n\n' %(count,residual))

#     psiOutAnderson = scipyAnderson(PowerIteration, psiIn,f_tol=1e-12,verbose=True)

    
    show_options(solver='root', method='anderson', disp=True)
#     options={'fatol':1e-12, 'disp':True}
    jacobianOptions={'alpha':1.0, 'M':5, 'w0':0.01}
    options={'fatol':1e-12, 'line_search':None, 'disp':True, 'jac_options':jacobianOptions}
#     options={'fatol':1e-12, 'maxiter':10, 'disp':True, 'jac_options':jacobianOptions}
    print(options)
    method='anderson'
#     options={'verbose':True}
    psiOutAndersonRoot = scipyRoot(PowerIteration, psiIn, args=('hello','hi'), method=method, options=options)
#         print(psiOutAndersonRoot.__dir__())
#         print(psiOutAndersonRoot.success)
#         print(psiOutAndersonRoot.message)
    try:
        print(method + ' number of evals: ', psiOutAndersonRoot.nit)
    except Exception:
        print(method + ' number of evals: ', psiOutAndersonRoot.nfev)
        
        
def precond(psi):      
    global A
    print(A)
    D = np.diag(A)
    print(D)
    Dinv = np.linalg.inv(D)
    print(Dinv)
    return np.dot(Dinv,psi)
        
def testKrylovOptions(N):
    global A, psi1, weights
    if N==3:
        A = np.array( [ [1,2,0.15],
                        [0,0.5,-0.3],
                        [0,0,1.0]
                    ] )
        psiIn=np.array([1,2,3])
    else:
        weights = np.ones(N)
        A = np.random.rand(N,N)
        A = A+A.T
        psiIn = np.random.rand(N)
    
    
    jac = BroydenFirst()
    kjac = KrylovJacobian(inner_M=InverseJacobian(jac))    
    method='krylov'
    
    show_options(solver='root', method=method, disp=True)
#     options={'fatol':1e-12, 'disp':True}
#     jacobianOptions={'method':'lgmres','inner_M':kjac, 'inner_maxiter':500, 'outer_k':3}
#     jacobianOptions={'method':'lgmres', 'inner_maxiter':500, 'outer_k':3}
#     jacobianOptions={'method':'bicgstab', 'inner_M':precond}
    jacobianOptions={'method':'bicgstab', 'inner_M':jac}
    jacobianOptions={'method':'bicgstab'}
    options={'fatol':1e-12, 'disp':True, 'maxiter':500, 'jac_options':jacobianOptions}

    
#     options={'verbose':True}
    psiOutAndersonRoot = scipyRoot(PowerIteration, psiIn, method=method, options=options)
#         print(psiOutAndersonRoot.__dir__())
#         print(psiOutAndersonRoot.success)
#         print(psiOutAndersonRoot.message)
    try:
        print(method + ' number of evals: ', psiOutAndersonRoot.nit)
    except Exception:
        print(method + ' number of function evals: ', psiOutAndersonRoot.nfev)
        print(method + ' number of jacobian evals: ', psiOutAndersonRoot.njev)
        
        
        
def testBroydenOptions(N):
    global A, psi1, weights
    if N==3:
        A = np.array( [ [1,0,0],
                        [0,0.5,0],
                        [0,0,0.99]
                    ] )
    else:
        weights = np.ones(N)
        A = np.random.rand(N,N)
        A = A+A.T
    
    
    
    psiIn = np.random.rand(N)
    

    method='broyden1'
    show_options(solver='root', method=method, disp=True)
#     options={'fatol':1e-12, 'disp':True}
    jacobianOptions={'alpha':1.0}
    options={'fatol':1e-12, 'line_search':None, 'disp':True, 'maxiter':500, 'jac_options':jacobianOptions}

    
#     options={'verbose':True}
    psiOutAndersonRoot = scipyRoot(PowerIteration, psiIn, method=method, options=options)
#         print(psiOutAndersonRoot.__dir__())
#         print(psiOutAndersonRoot.success)
#         print(psiOutAndersonRoot.message)
    try:
        print(method + ' number of evals: ', psiOutAndersonRoot.nit)
    except Exception:
        print(method + ' number of evals: ', psiOutAndersonRoot.nfev)
        
        
        
        
        

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
    psi1 = scipyAnderson(PowerIteration,psiIn,M=10,w0=0.01,tol_norm=np.linalg.norm,f_tol=1e-7,verbose=True)
    print('Rayleigh Quotient: ', np.dot( psi1, np.dot(A,psi1)) / np.dot(psi1,psi1))
    print('Difference: ', np.dot( psi1, np.dot(A,psi1)) / np.dot(psi1,psi1)-eig1)
    print()
    print()
    
    global psiOrth
    psiOrth = np.copy(psi1)
     
    psiIn = np.random.rand(N)
    psiIn /= np.linalg.norm(psiIn)
    
    residual=1
    count=1
    
#     d = {psiOrth":psiOrth}

    # Preprocessing
    while ( (residual > 1e-1) and (count<10000) ):
        r = PowerIterationOptional(psiIn)
        psiIn += r
        residual = np.linalg.norm(r)
#         print('Iteration %i: Residual = %1.2e' %(count,residual))
        count+=1
#     print('Rayleigh Quotient: ', np.dot( psi2, np.dot(A,psi2)) / np.dot(psi2,psi2))
    
    
    print('Using Anderson for PowerIterationOptional...')
#     Fkw = {"psiOrth":psiOrth}
#     AndersonKW = {"M":10, "w0":0.01,"tol_norm":np.linalg.norm,"f_tol":1e-7,"verbose":True }
    psi2 = scipyAnderson(PowerIterationOptional,psiIn,M=10, w0=0.01,tol_norm=np.linalg.norm,f_tol=1e-7,verbose=True)
    print('Rayleigh Quotient: ', np.dot( psi2, np.dot(A,psi2)) / np.dot(psi2,psi2))
    print('Difference: ', np.dot( psi2, np.dot(A,psi2)) / np.dot(psi2,psi2)-eig2)
    print('Used %i iterations in preprocessing.' %count)
#     print('psi1 = ', psi1)
#     print('psi2 = ', psi2)
#     print('Overlap: ', np.dot(psi1,psi2))


if __name__=="__main__":
#     testRootfinders(3)
#     testRootfinderOptions(3)
#     testKrylovOptions(3)
    testAndersonOptions(30)
#     testBroydenOptions(3)
#     test(10)