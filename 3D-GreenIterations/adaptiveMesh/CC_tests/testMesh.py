'''
Created on Jun 25, 2018

@author: nathanvaughn
'''
import sys
sys.path.append('../src/dataStructures')
sys.path.append('../src/utilities')
import itertools
import numpy as np
ThreeByThreeByThree = [element for element in itertools.product(range(3),range(3),range(3))]

from TreeStruct_CC import Tree



def exportMeshForParaview(xmin,xmax,px,ymin,ymax,py,zmin,zmax,pz,minLevels, maxLevels, divideCriterion, divideParameter,coordinateFile):    
#     tree = Tree(xmin,xmax,px,ymin,ymax,py,zmin,zmax,pz,coordinateFile=coordinateFile)
#     tree.buildTree( minLevels, maxLevels, divideCriterion, divideParameter, printTreeProperties=True)
#     tree.exportMeshVTK('/Users/nathanvaughn/Desktop/hydrogenMolecule.vtk')
#     tree.exportGridpoints('/Users/nathanvaughn/Desktop/hydrogenMolecule')
    
    tree = Tree(xmin,xmax,px,ymin,ymax,py,zmin,zmax,pz,nElectrons=14,nOrbitals=10,coordinateFile=coordinateFile)
    tree.buildTree( minLevels, maxLevels, divideCriterion, divideParameter, printTreeProperties=True)
    tree.exportGridpoints('/Users/nathanvaughn/Desktop/carbonMonoxide_max14')

    print('Mesh Exported.')
    
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
    

            

if __name__ == "__main__":
    timingTestsForOrbitalInitializations(domain=20,order=5,
                          minDepth=3, maxDepth=20, divideCriterion='LW5', 
                          divideParameter=1500,inputFile='../src/utilities/molecularConfigurations/oxygenAtomAuxiliary.csv')
    
    
#     exportMeshForParaview(xmin=-10, xmax=10,px=4,
#                           ymin=-10, ymax=10,py=4,
#                           zmin=-10, zmax=10,pz=4,
#                           minLevels=3, maxLevels=20, divideCriterion='LW1', 
#                           divideParameter=200,coordinateFile='../src/utilities/molecularConfigurations/diatomic_example.csv')
    
    
    
    