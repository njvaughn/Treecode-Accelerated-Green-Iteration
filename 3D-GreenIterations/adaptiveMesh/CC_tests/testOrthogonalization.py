'''
Created on Jun 25, 2018

@author: nathanvaughn
'''
import sys
sys.path.append('../src/dataStructures')
sys.path.append('../src/utilities')
import itertools
import time
import numpy as np
from numba import jit

ThreeByThreeByThree = [element for element in itertools.product(range(3),range(3),range(3))]

from TreeStruct_CC import Tree


@jit(nopython=True,parallel=True)
def modifiedGramSchrmidt(V,weights):
    n,k = np.shape(V)
    U = np.zeros_like(V)
    U[:,0] = V[:,0] / np.dot(V[:,0],V[:,0]*weights)
    for i in range(1,k):
        U[:,i] = V[:,i]
        for j in range(i):
#             print('Orthogonalizing %i against %i' %(i,j))
            U[:,i] -= (np.dot(U[:,i],U[:,j]*weights) / np.dot(U[:,j],U[:,j]*weights))*U[:,j]
        U[:,i] /= np.dot(U[:,i],U[:,i]*weights)
        
    return U

# @jit(nopython=True,parallel=True)
@jit(parallel=True)
def modifiedGramSchrmidt_singleOrbital(V,weights,targetOrbital):
    n,k = np.shape(V)
    U = V[:,targetOrbital]
    for j in range(targetOrbital):
        print('Orthogonalizing %i against %i' %(targetOrbital,j))
        U -= (np.dot(V[:,targetOrbital],V[:,j]*weights) / np.dot(V[:,j],V[:,j]*weights))*V[:,j]
#         U -= np.dot(V[:,targetOrbital],V[:,j]*weights) *V[:,j]
        U /= np.sqrt( np.dot(U,U*weights) )
        
    return U

    

def timingTestsForOrbitalOrthogonalizations(domain,order,minDepth, maxDepth, divideCriterion, divideParameter,inputFile):
    [coordinateFile, referenceEigenvaluesFile, DummyOutputFile] = np.genfromtxt(inputFile,dtype="|U100")[:3]
    [Eband, Ekinetic, Eexchange, Ecorrelation, Eelectrostatic, Etotal, gaugeShift] = np.genfromtxt(inputFile)[3:]
    
    
    print('Reading atomic coordinates from: ', coordinateFile)
    atomData = np.genfromtxt(coordinateFile,delimiter=',',dtype=float)
    if np.shape(atomData)==(5,):
        nElectrons = atomData[3]
    else:
        nElectrons = 0
        for i in range(len(atomData)):
            nElectrons += atomData[i,3]
    
    nOrbitals = int( np.ceil(nElectrons/2) + 1 )  
    occupations = 2*np.ones(nOrbitals)
    occupations[-1] = 0
    
    print([coordinateFile, nElectrons, nOrbitals, 
     Etotal, Eexchange, Ecorrelation, Eband, gaugeShift])
    tree = Tree(-domain,domain,order,-domain,domain,order,-domain,domain,order,nElectrons,nOrbitals,maxDepthAtAtoms=maxDepth,gaugeShift=gaugeShift,
                coordinateFile=coordinateFile,inputFile=inputFile)#, iterationOutFile=outputFile)

    
    tree.buildTree( minLevels=minDepth, maxLevels=maxDepth, initializationType='atomic',divideCriterion=divideCriterion, divideParameter=divideParameter, printTreeProperties=True,onlyFillOne=False)
    sources = tree.extractPhi(0)
    
    ## Copy orbitals to a local array.  For timing, this will not count against the truly external orthogonalization
    orbitals = np.zeros((len(sources),tree.nOrbitals))
    for m in range(nOrbitals):
        # fill in orbitals
        targets = tree.extractPhi(m)
        orbitals[:,m] = np.copy(targets[:,3])
    weights = np.copy(targets[:,5])
    
#     phiC0 = modifiedGramSchrmidt_singleOrbital(orbitals, weights, 0)
#     phiC1 = modifiedGramSchrmidt_singleOrbital(orbitals, weights, 1)
#     phiC2 = modifiedGramSchrmidt_singleOrbital(orbitals, weights, 2)



    testOrbital = 6
    
    
    start = time.time()
    phiC3 = modifiedGramSchrmidt_singleOrbital(orbitals, weights, testOrbital)
    trulyExternalTime = time.time()-start
    print('Time for truly external orthogonalization: ', trulyExternalTime)
    
    
    
    start = time.time()
    tree.orthonormalizeOrbitals(targetOrbital=testOrbital, external=False)
    internalTime = time.time()-start
    
    print('\n\nTime for internal orthogonalization: ', internalTime)
    
    sources = tree.extractPhi(0)
    phiA0 = sources[:,3]
    sources = tree.extractPhi(testOrbital)
    phiA3 = sources[:,3]
    
    start = time.time()
    tree.orthonormalizeOrbitals(targetOrbital=testOrbital, external=True)
    externalTime = time.time()-start

    print('Time for external orthogonalization: ', externalTime)


    sources = tree.extractPhi(0)
    phiB0 = sources[:,3]
    sources = tree.extractPhi(testOrbital)
    phiB3 = sources[:,3]
    
    
    
    
    
    
    
    

    
    
#     print('Max diff between internal and external: ', np.max( np.abs(phiA0 - phiB0 )))
    print('Max diff between internal and external: ', np.max( np.abs(phiA3 - phiB3 )))
    print('Max diff between internal and truly external: ', np.max( np.abs(phiA3 - phiC3 )))
    
    speedup = externalTime/trulyExternalTime
    print('Speedup: ', speedup)
            

if __name__ == "__main__":

    timingTestsForOrbitalOrthogonalizations(domain=20,order=3,
                          minDepth=3, maxDepth=20, divideCriterion='LW5', 
#                           divideParameter=500,inputFile='../src/utilities/molecularConfigurations/oxygenAtomAuxiliary.csv')
                        divideParameter=500,inputFile='../src/utilities/molecularConfigurations/carbonMonoxideAuxiliary.csv')
    

    
    
