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



def exportMeshForParaview(xmax,px,minLevels, maxLevels, divideCriterion, divideParameter):    
    tree = Tree(-xmax,xmax,px,-xmax,xmax,px,-xmax,xmax,px)
    tree.buildTree( minLevels, maxLevels, divideCriterion, divideParameter, printTreeProperties=True)
    
#     tree.exportMeshMidpointsForParaview('/Users/nathanvaughn/Desktop/meshTesting.csv')
#     tree.exportMeshQuadpointsForParaview('/Users/nathanvaughn/Desktop/quadTesting.csv')
    tree.exportMeshVerticesForParaview('/Users/nathanvaughn/Desktop/verticesTesting.csv')
    print('Mesh Exported.')
    
    

            

if __name__ == "__main__":
    exportMeshForParaview(xmax=10,px=5,minLevels=2, maxLevels=15, divideCriterion='LW1', divideParameter=10)