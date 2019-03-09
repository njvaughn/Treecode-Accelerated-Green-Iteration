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
import matplotlib.pyplot as plt

from meshUtilities import *
ThreeByThreeByThree = [element for element in itertools.product(range(3),range(3),range(3))]

from TreeStruct_CC import Tree


def exportMeshForTreecodeTesting(domain,order,minDepth, maxDepth, depthAtAtoms, divideCriterion, divideParameter1, divideParameter2, divideParameter3, divideParameter4, inputFile,
                                 smoothingEpsilon=0.0):

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
    
    nOrbitals = int( np.ceil(nElectrons/2)+1)
    print('nElectrons = ', nElectrons)
    print('nOrbitals  = ', nOrbitals)
    print([coordinateFile, Etotal, Eexchange, Ecorrelation, Eband, gaugeShift])
    tree = Tree(-domain,domain,order,-domain,domain,order,-domain,domain,order,nElectrons,nOrbitals,maxDepthAtAtoms=depthAtAtoms,minDepth=minDepth,gaugeShift=gaugeShift,
                coordinateFile=coordinateFile,smoothingEps=smoothingEpsilon,inputFile=inputFile)#, iterationOutFile=outputFile)
    
     
     

    
    print('max depth ', maxDepth)

#     tree.buildTree( maxLevels=maxDepth, initializationType='atomic',divideCriterion=divideCriterion, divideParameter=divideParameter, printTreeProperties=True,onlyFillOne=False)
    tree.buildTree( maxLevels=maxDepth, initializationType='atomic',divideCriterion=divideCriterion, 
                    divideParameter1=divideParameter1, divideParameter2=divideParameter2, divideParameter3=divideParameter3, divideParameter4=divideParameter4, 
                    printTreeProperties=True,onlyFillOne=False)
    
#     sourcesTXT = '../examples/S%ipy.txt' %tree.numberOfGridpoints
#     targetsTXT = '../examples/T%ipy.txt' %tree.numberOfGridpoints
    
#     sourcesTXT = '/Users/nathanvaughn/Documents/GitHub/hybrid-gpu-treecode/examplesOxygenAtom/S%ipy.txt' %tree.numberOfGridpoints
    sourcesTXT = '/Users/nathanvaughn/Desktop/S%ipy.txt' %tree.numberOfGridpoints
    targetsTXT = '/Users/nathanvaughn/Desktop/T%ipy.txt' %tree.numberOfGridpoints
    
    Sources = tree.extractLeavesDensity()
    Targets = tree.extractLeavesDensity()

#     print(Targets[0,:])
    print(Targets[0:3,0:4])
    
    # Save as .txt files
    np.savetxt(sourcesTXT, Sources)
    np.savetxt(targetsTXT, Targets[:,0:4])

    print('Meshes Exported.')    


def exportMeshForParaview(domain,order,minDepth, maxDepth, additionalDepthAtAtoms, divideCriterion, 
                          divideParameter1, divideParameter2=0.0, divideParameter3=0.0, divideParameter4=0.0, 
                          smoothingEpsilon=0.0, 
                          inputFile='', outputFile=''):    
    
    
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
    
#     nOrbitals = int( np.ceil(nElectrons/2))
    nOrbitals = int( np.ceil(nElectrons/2)+1)

    if inputFile=='../src/utilities/molecularConfigurations/benzeneAuxiliary.csv':
        nOrbitals = 30
    print('nElectrons = ', nElectrons)
    print('nOrbitals  = ', nOrbitals)
    print([coordinateFile, Etotal, Eexchange, Ecorrelation, Eband, gaugeShift])
    tree = Tree(-domain,domain,order,-domain,domain,order,-domain,domain,order,nElectrons,nOrbitals,additionalDepthAtAtoms=additionalDepthAtAtoms,minDepth=minDepth,gaugeShift=gaugeShift,
                coordinateFile=coordinateFile,smoothingEps=smoothingEpsilon,inputFile=inputFile)#, iterationOutFile=outputFile)

    
    print('max depth ', maxDepth)
    tree.buildTree( maxLevels=maxDepth, initializationType='atomic',divideCriterion=divideCriterion, 
                    divideParameter1=divideParameter1, divideParameter2=divideParameter2, divideParameter3=divideParameter3, divideParameter4=divideParameter4, 
                    printTreeProperties=True,onlyFillOne=False)
    
    print(tree.levelCounts)
    print()
    print(tree.levelCounts.keys())
    print()
    print(tree.levelCounts.values())
    plt.bar(list(tree.levelCounts.keys()),list(tree.levelCounts.values()) )
    plt.xlabel('Refinement Depth')
    plt.ylabel('Number of Cells')
    if divideCriterion=='Krasny':
        plt.title('Mesh Type: 4 parameters (%1.2f,%1.2f,%1.2f,%1.2f)' %(divideParameter1,divideParameter2,divideParameter3,divideParameter4))
    elif divideCriterion=='LW5':
        plt.title('Mesh Type: LW5 - %1.2f' %(divideParameter1))
    plt.show()
#     tree.sortOrbitalsAndEnergies(order = [5,0,6,1,2,8,9,3,4,7])
    
#     tree.exportGridpoints('/Users/nathanvaughn/Desktop/meshTests/Biros/Beryllium_order5_1em4')
    tree.exportGridpoints(outputFile)
#     tree.orthonormalizeOrbitals()
#     tree.exportGridpoints('/Users/nathanvaughn/Desktop/meshTests/CO_afterOrth')

    print('Meshes Exported.')
    
def timingTestsForOrbitalInitializations(domain,order,minDepth, maxDepth, depthAtAtoms, divideCriterion, divideParameter,inputFile):
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
    

def meshDistributions(domain,order,minDepth, maxDepth, additionalDepthAtAtoms, divideCriterion, 
                          divideParameter1, divideParameter2=0.0, divideParameter3=0.0, divideParameter4=0.0, 
                          smoothingEpsilon=0.0, base=1.0,
                          inputFile=''):    
    
    
    divideParameter1 *= base
    divideParameter2 *= base
    divideParameter3 *= base
    divideParameter4 *= base
    
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
    
#     nOrbitals = int( np.ceil(nElectrons/2))
    nOrbitals = int( np.ceil(nElectrons/2)+1)

    if inputFile=='../src/utilities/molecularConfigurations/benzeneAuxiliary.csv':
        nOrbitals = 30
    print('nElectrons = ', nElectrons)
    print('nOrbitals  = ', nOrbitals)
    print([coordinateFile, Etotal, Eexchange, Ecorrelation, Eband, gaugeShift])
    tree = Tree(-domain,domain,order,-domain,domain,order,-domain,domain,order,nElectrons,nOrbitals,additionalDepthAtAtoms=additionalDepthAtAtoms,minDepth=minDepth,gaugeShift=gaugeShift,
                coordinateFile=coordinateFile,smoothingEps=smoothingEpsilon,inputFile=inputFile)#, iterationOutFile=outputFile)

    
    print('max depth ', maxDepth)
    tree.buildTree( maxLevels=maxDepth, initializationType='atomic',divideCriterion=divideCriterion, 
                    divideParameter1=divideParameter1, divideParameter2=divideParameter2, divideParameter3=divideParameter3, divideParameter4=divideParameter4, 
                    printTreeProperties=True,onlyFillOne=False)
    
    print(tree.levelCounts)
    print()
    print(tree.levelCounts.keys())
    print()
    print(tree.levelCounts.values())
    maxHeight = np.max(list(tree.levelCounts.values()))
    plt.figure()
    plt.bar(list(tree.levelCounts.keys()),list(tree.levelCounts.values()) )
    plt.xlabel('Refinement Depth')
    plt.ylabel('Number of Cells')
    if divideCriterion=='Krasny':
        plt.title('Mesh Type: 4 parameters (%1.2f,%1.2f,%1.2f,%1.2f)' %(divideParameter1,divideParameter2,divideParameter3,divideParameter4))
    if divideCriterion=='Nathan':
        plt.title('Mesh Type: 2 parameters (%1.2f,%1.2f)' %(divideParameter1,divideParameter2))
    elif divideCriterion=='LW5':
        plt.title('Mesh Type: LW5 - %1.2f' %(divideParameter1))
        
    fig, axes = plt.subplots(2, 1)
    axes[0].bar(list(tree.criteria1.keys()),list(tree.criteria1.values()) )
    axes[1].bar(list(tree.criteria2.keys()),list(tree.criteria2.values()) )
#     axes[1,0].bar(list(tree.criteria3.keys()),list(tree.criteria3.values()) )
#     axes[1,1].bar(list(tree.criteria4.keys()),list(tree.criteria4.values()) )
     
    axes[0].set_title('Vext*sqrt(rho) Integral')
    axes[1].set_title('sqrt(rho) integral')
#     axes[1,0].set_title('rho integral')
#     axes[1,1].set_title('Vext Variation')
     
    axes[0].set_ylim([0, int(1.1*maxHeight)])
    axes[1].set_ylim([0, int(1.1*maxHeight)])
#     axes[1,0].set_ylim([0, int(1.1*maxHeight)])
#     axes[1,1].set_ylim([0, int(1.1*maxHeight)])
    plt.tight_layout()
  
    plt.show()
    
    
def plot_LW_density():
    r = np.linspace(1e-1,1,1000)
    
    
    
    plt.figure()
    for N in [3,4,5]:
        density = np.zeros(len(r))
        if N==3:
            A = 648.82
            c = [52, -102, 363, 1416, 4164, 5184, 2592]
        elif N == 4:
            A = 1797.9
            c = [423,-1286,2875,16506,79293,292512,611136,697320, 348660]
        elif N==5:
            A = 3697.1
            c = [2224, -9018, 16789, 117740, 733430, 3917040, 16879920, 49186500, 91604250, 100516500, 50258250]
        for k in range(0,2*N+1):
            density += c[k] * r**(-k)
        
        density *= np.exp(-2*r)
        density = 1000/A*density**(3/(2*N+3))
        
        
        plt.plot(r,density, label = 'N = %i'%N)
    plt.legend()
    plt.xlabel('Distance from nucleus')
    plt.ylabel('LW Mesh Density')
    plt.show()

if __name__ == "__main__":
    
#     plot_LW_density()
    
#     meshDistributions(domain=20,order=3,
#                         minDepth=3, maxDepth=20, additionalDepthAtAtoms=3, divideCriterion='Nathan', 
#                         divideParameter1=0.1/5, divideParameter2=10.1/1, divideParameter3=100, divideParameter4=100,
#                         smoothingEpsilon=0.0,base=1.0, inputFile='../src/utilities/molecularConfigurations/berylliumAuxiliary.csv')
    
    
#     timingTestsForOrbitalInitializations(domain=20,order=5,
#                           minDepth=3, maxDepth=20, divideCriterion='LW5', 
#                           divideParameter=1500,inputFile='../src/utilities/molecularConfigurations/oxygenAtomAuxiliary.csv')


#     timingTestsForOrbitalOrthogonalizations(domain=20,order=4,
#                           minDepth=3, maxDepth=20, divideCriterion='LW5', 
#                           divideParameter=500,inputFile='../src/utilities/molecularConfigurations/oxygenAtomAuxiliary.csv')
    
    # param1: wavefunction variation
    # param2: wavefunction integral
    # param3: density integral   
    # param4: Vext integral   
    
    exportMeshForParaview(domain=20,order=5,
                        minDepth=3, maxDepth=15, additionalDepthAtAtoms=0, divideCriterion='Krasny_density', 
                        divideParameter1=100, divideParameter2=1000.1, divideParameter3=1000.2, divideParameter4=20000,
                        smoothingEpsilon=0.0,inputFile='../src/utilities/molecularConfigurations/berylliumAuxiliary.csv', 
                        outputFile='/Users/nathanvaughn/Desktop/meshTests/oxygen/LW5_1500')
    


#     exportMeshForTreecodeTesting(domain=20,order=7,
#                         minDepth=3, maxDepth=20, depthAtAtoms=13, divideCriterion='LW5', 
#                         divideParameter1=2000, divideParameter2=100, divideParameter3=0.03, divideParameter4=5000,
#                         inputFile='../src/utilities/molecularConfigurations/carbonMonoxideAuxiliary.csv')

#                         divideParameter=1e-5,inputFile='../src/utilities/molecularConfigurations/hydrogenMoleculeAuxiliary.csv')
#                         divideParameter1=1.0, divideParameter2=1.0,inputFile='../src/utilities/molecularConfigurations/oxygenAtomAuxiliary.csv')
#                         divideParameter1=500,inputFile='../src/utilities/molecularConfigurations/oxygenAtomAuxiliary.csv')
#                         divideParameter=1e-3,inputFile='../src/utilities/molecularConfigurations/berylliumAuxiliary.csv')
#                         divideParameter=1.25e-3,inputFile='../src/utilities/molecularConfigurations/carbonMonoxideAuxiliary.csv')
     
    
