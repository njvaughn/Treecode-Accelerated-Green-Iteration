'''
Created on Jun 25, 2018

@author: nathanvaughn
'''
import sys
sys.path.append('../src/dataStructures')
sys.path.append('../src/utilities')
import itertools
ThreeByThreeByThree = [element for element in itertools.product(range(3),range(3),range(3))]

from TreeStruct_CC import Tree



def uniformRefinement(xmin,xmax,px,ymin,ymax,py,zmin,zmax,pz,minLevels, maxLevels, divideCriterion, divideParameter,coordinateFile):    
#     tree = Tree(xmin,xmax,px,ymin,ymax,py,zmin,zmax,pz,coordinateFile=coordinateFile)
#     tree.buildTree( minLevels, maxLevels, divideCriterion, divideParameter, printTreeProperties=True)
#     tree.exportMeshVTK('/Users/nathanvaughn/Desktop/hydrogenMolecule.vtk')
#     tree.exportGridpoints('/Users/nathanvaughn/Desktop/hydrogenMolecule')
    
    tree = Tree(xmin,xmax,px,ymin,ymax,py,zmin,zmax,pz,nElectrons=8,nOrbitals=5,coordinateFile=coordinateFile)
    tree.buildTree( minLevels, maxLevels, divideCriterion, divideParameter, printTreeProperties=True)
    
    print()
    tree.uniformlyRefine()
    tree.uniformlyRefine()
#     tree.exportGridpoints('/Users/nathanvaughn/Desktop/oxygenAtom')
# 
#     print('Mesh Exported.')
    
    

            

if __name__ == "__main__":
    uniformRefinement(xmin=-20, xmax=20,px=3,
                          ymin=-20, ymax=20,py=3,
                          zmin=-20, zmax=20,pz=3,
                          minLevels=2, maxLevels=20, divideCriterion='LW3', 
#                           divideParameter=100,coordinateFile='../src/utilities/molecularConfigurations/hydrogenMolecule.csv')
#                           divideParameter=100,coordinateFile='../src/utilities/molecularConfigurations/berylliumAtom.csv')
                          divideParameter=100,coordinateFile='../src/utilities/molecularConfigurations/oxygenAtom.csv')
    
    

    
    