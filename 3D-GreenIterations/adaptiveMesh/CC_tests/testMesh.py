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



def exportMeshForParaview(xmin,xmax,px,ymin,ymax,py,zmin,zmax,pz,minLevels, maxLevels, divideCriterion, divideParameter,inputFile):    
    tree = Tree(xmin,xmax,px,ymin,ymax,py,zmin,zmax,pz,inputFile)
    tree.exportMeshVTK('/Users/nathanvaughn/Desktop/coaxial.vtk')
#     tree.buildTree( minLevels, maxLevels, divideCriterion, divideParameter, printTreeProperties=True)
#     tree.exportMeshVTK('/Users/nathanvaughn/Desktop/coaxial.vtk')
    
#     tree.exportMeshMidpointsForParaview('/Users/nathanvaughn/Desktop/meshTesting.csv')
#     tree.exportMeshQuadpointsForParaview('/Users/nathanvaughn/Desktop/quadTesting.csv')
#     tree.exportMeshVerticesForParaview('/Users/nathanvaughn/Desktop/verticesTesting.csv')
#     tree.exportMeshVTK('/Users/nathanvaughn/Desktop/verticesVTK_quadrupleAtom.vtk')
#     tree.exportMeshVTK('/Users/nathanvaughn/Desktop/1Dchain.vtk')
#     for atom in tree.atoms:
#         print('atom: x,y,z: ', atom.x, atom.y, atom.z)
    print('Mesh Exported.')
    
    

            

if __name__ == "__main__":
    exportMeshForParaview(xmin=-10, xmax=10,px=3,
                          ymin=-10, ymax=10,py=3,
                          zmin=-10, zmax=10,pz=3,
                          minLevels=2, maxLevels=20, divideCriterion='LW1', 
                          divideParameter=500,inputFile='../src/utilities/molecularConfigurations/diatomic_example.csv')
    
    
    
    
    
    