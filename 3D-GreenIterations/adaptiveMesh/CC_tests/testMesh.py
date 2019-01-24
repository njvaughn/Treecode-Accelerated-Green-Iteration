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
ThreeByThreeByThree = [element for element in itertools.product(range(3),range(3),range(3))]

from TreeStruct_CC import Tree



def exportMeshForParaview(domain,order,minDepth, maxDepth, divideCriterion, divideParameter,inputFile):    
    
    
#     [coordinateFile, DummyOutputFile] = np.genfromtxt(inputFile,dtype="|U100")[:2]
    [coordinateFile, referenceEigenvaluesFile, DummyOutputFile] = np.genfromtxt(inputFile,dtype="|U100")[:3]
    [Eband, Ekinetic, Eexchange, Ecorrelation, Eelectrostatic, Etotal, gaugeShift] = np.genfromtxt(inputFile)[3:]
#     nElectrons = int(nElectrons)
#     nOrbitals = int(nOrbitals)

    print('Reading atomic coordinates from: ', coordinateFile)
    atomData = np.genfromtxt(coordinateFile,delimiter=',',dtype=float)
    if np.shape(atomData)==(5,):
        nElectrons = atomData[3]
    else:
        nElectrons = 0
        for i in range(len(atomData)):
            nElectrons += atomData[i,3]
    
    nOrbitals = int( np.ceil(nElectrons/2)+1)
    print('nElectrons = ', nElectrons)
    print('nOrbitals  = ', nOrbitals)
    print([coordinateFile, Etotal, Eexchange, Ecorrelation, Eband, gaugeShift])
    tree = Tree(-domain,domain,order,-domain,domain,order,-domain,domain,order,nElectrons,nOrbitals,maxDepthAtAtoms=maxDepth,minDepth=minDepth,gaugeShift=gaugeShift,
                coordinateFile=coordinateFile,inputFile=inputFile)#, iterationOutFile=outputFile)

    
    print('max depth ', maxDepth)
    tree.buildTree( maxLevels=maxDepth, initializationType='atomic',divideCriterion=divideCriterion, divideParameter=divideParameter, printTreeProperties=True,onlyFillOne=False)
#     tree.sortOrbitalsAndEnergies(order = [5,0,6,1,2,8,9,3,4,7])
    
#     tree.exportGridpoints('/Users/nathanvaughn/Desktop/meshTests/Biros/Beryllium_order5_1em4')
    tree.exportGridpoints('/Users/nathanvaughn/Desktop/meshTests/Biros/CO_order5_1em3')
#     tree.orthonormalizeOrbitals()
#     tree.exportGridpoints('/Users/nathanvaughn/Desktop/meshTests/CO_afterOrth')

    print('Meshes Exported.')
    
def timingTestsForOrbitalInitializations(domain,order,minDepth, maxDepth, divideCriterion, divideParameter,inputFile):
    [coordinateFile, DummyOutputFile] = np.genfromtxt(inputFile,dtype="|U100")[:2]
    [nElectrons, nOrbitals, Eband, Ekinetic, Eexchange, Ecorrelation, Eelectrostatic, Etotal, gaugeShift] = np.genfromtxt(inputFile)[2:]
    nElectrons = int(nElectrons)
    nOrbitals = int(nOrbitals)
    
    print([coordinateFile, nElectrons, nOrbitals, 
     Etotal, Eexchange, Ecorrelation, Eband, gaugeShift])
    tree = Tree(-domain,domain,order,-domain,domain,order,-domain,domain,order,nElectrons,nOrbitals,maxDepthAtAtoms=maxDepth,gaugeShift=gaugeShift,
                coordinateFile=coordinateFile,inputFile=inputFile)#, iterationOutFile=outputFile)

    
    print('max depth ', maxDepth)
    tree.buildTree( minLevels=minDepth, maxLevels=maxDepth, initializationType='atomic',divideCriterion=divideCriterion, divideParameter=divideParameter, printTreeProperties=True,onlyFillOne=False)
    
#     afterInternal = tree.extractLeavesDensity()
#     print('Max density = ', max(afterInternal[:,3]))
#     tree.initializeDensityFromAtomicDataExternally()
#     afterExternal = tree.extractLeavesDensity()
#     print('Max diff between internal and external: ', np.max( np.abs(afterInternal[:,3] - afterExternal[:,3] )))



    afterInternal0 = tree.extractPhi(0)
    afterInternal2 = tree.extractPhi(2)
    
    tree.initializeOrbitalsFromAtomicDataExternally()
    
    afterExternal0 = tree.extractPhi(0)
    afterExternal2 = tree.extractPhi(2)
    
    print('Max diff between internal0 and external0: ', np.max( np.abs(afterInternal0[:,3] - afterExternal0[:,3] )))
    print('Max diff between internal2 and external2: ', np.max( np.abs(afterInternal2[:,3] - afterExternal2[:,3] )))
    

def timingTestsForOrbitalOrthogonalizations(domain,order,minDepth, maxDepth, divideCriterion, divideParameter,inputFile):
    [coordinateFile, DummyOutputFile] = np.genfromtxt(inputFile,dtype="|U100")[:2]
    [nElectrons, nOrbitals, Eband, Ekinetic, Eexchange, Ecorrelation, Eelectrostatic, Etotal, gaugeShift] = np.genfromtxt(inputFile)[2:]
    nElectrons = int(nElectrons)
    nOrbitals = int(nOrbitals)
    
    print([coordinateFile, nElectrons, nOrbitals, 
     Etotal, Eexchange, Ecorrelation, Eband, gaugeShift])
    tree = Tree(-domain,domain,order,-domain,domain,order,-domain,domain,order,nElectrons,nOrbitals,maxDepthAtAtoms=maxDepth,gaugeShift=gaugeShift,
                coordinateFile=coordinateFile,inputFile=inputFile)#, iterationOutFile=outputFile)

    
    tree.buildTree( minLevels=minDepth, maxLevels=maxDepth, initializationType='atomic',divideCriterion=divideCriterion, divideParameter=divideParameter, printTreeProperties=True,onlyFillOne=False)
    
    start = time.time()
    tree.orthonormalizeOrbitals(targetOrbital=3, external=False)
    internalTime = time.time()-start
    
    print('\n\nTime for internal orthogonalization: ', internalTime)
    
    sources = tree.extractPhi(0)
    phiA0 = sources[:,3]
    sources = tree.extractPhi(3)
    phiA3 = sources[:,3]
    
    start = time.time()
    tree.orthonormalizeOrbitals(targetOrbital=3, external=True)
    externalTime = time.time()-start

    print('Time for external orthogonalization: ', externalTime)


    sources = tree.extractPhi(0)
    phiB0 = sources[:,3]
    sources = tree.extractPhi(3)
    phiB3 = sources[:,3]
    
    print('Max diff between internal and external: ', np.max( np.abs(phiA0 - phiB0 )))
    print('Max diff between internal and external: ', np.max( np.abs(phiA3 - phiB3 )))
    

            

if __name__ == "__main__":
#     timingTestsForOrbitalInitializations(domain=20,order=5,
#                           minDepth=3, maxDepth=20, divideCriterion='LW5', 
#                           divideParameter=1500,inputFile='../src/utilities/molecularConfigurations/oxygenAtomAuxiliary.csv')


#     timingTestsForOrbitalOrthogonalizations(domain=20,order=4,
#                           minDepth=3, maxDepth=20, divideCriterion='LW5', 
#                           divideParameter=500,inputFile='../src/utilities/molecularConfigurations/oxygenAtomAuxiliary.csv')
    
    
    exportMeshForParaview(domain=20,order=6,
                        minDepth=3, maxDepth=15, divideCriterion='Biros', 
#                         divideParameter=1e-5,inputFile='../src/utilities/molecularConfigurations/hydrogenMoleculeAuxiliary.csv')
                        divideParameter=5e-5,inputFile='../src/utilities/molecularConfigurations/oxygenAtomAuxiliary.csv')
#                         divideParameter=1e-3,inputFile='../src/utilities/molecularConfigurations/berylliumAuxiliary.csv')
#                         divideParameter=2e-0,inputFile='../src/utilities/molecularConfigurations/carbonMonoxideAuxiliary.csv')
    
    
    
#     orthonormalizeOrbitals