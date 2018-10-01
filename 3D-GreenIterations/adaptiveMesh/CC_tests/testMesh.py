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



def exportMeshForParaview(xmin,xmax,px,ymin,ymax,py,zmin,zmax,pz,minLevels, maxLevels, divideCriterion, divideParameter,coordinateFile):    
#     tree = Tree(xmin,xmax,px,ymin,ymax,py,zmin,zmax,pz,coordinateFile=coordinateFile)
#     tree.buildTree( minLevels, maxLevels, divideCriterion, divideParameter, printTreeProperties=True)
#     tree.exportMeshVTK('/Users/nathanvaughn/Desktop/hydrogenMolecule.vtk')
#     tree.exportGridpoints('/Users/nathanvaughn/Desktop/hydrogenMolecule')
    
    tree = Tree(xmin,xmax,px,ymin,ymax,py,zmin,zmax,pz,nElectrons=14,nOrbitals=10,coordinateFile=coordinateFile)
    tree.buildTree( minLevels, maxLevels, divideCriterion, divideParameter, printTreeProperties=True)
    tree.exportGridpoints('/Users/nathanvaughn/Desktop/carbonMonoxide_max14')

    print('Mesh Exported.')
    
    

            

if __name__ == "__main__":
    exportMeshForParaview(xmin=-20, xmax=20,px=3,
                          ymin=-20, ymax=20,py=3,
                          zmin=-20, zmax=20,pz=3,
                          minLevels=3, maxLevels=20, divideCriterion='LW3', 
#                           divideParameter=100,coordinateFile='../src/utilities/molecularConfigurations/hydrogenMolecule.csv')
#                           divideParameter=100,coordinateFile='../src/utilities/molecularConfigurations/berylliumAtom.csv')
                          divideParameter=1,coordinateFile='../src/utilities/molecularConfigurations/carbonMonoxide.csv')
    
    
#     exportMeshForParaview(xmin=-10, xmax=10,px=4,
#                           ymin=-10, ymax=10,py=4,
#                           zmin=-10, zmax=10,pz=4,
#                           minLevels=3, maxLevels=20, divideCriterion='LW1', 
#                           divideParameter=200,coordinateFile='../src/utilities/molecularConfigurations/diatomic_example.csv')
    
    
    
    