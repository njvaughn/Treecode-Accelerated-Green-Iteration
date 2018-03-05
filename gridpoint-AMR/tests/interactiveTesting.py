'''
Created on Feb 21, 2018

@author: nathanvaughn
'''
from Tree import Tree
from Cell import Cell
from Gridpoint import GridPoint
import numpy as np
from timer import Timer
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


    
if __name__ == "__main__":

    xmin = ymin = zmin = -10
    xmax = ymax = zmax = -xmin
    tree = Tree(xmin,xmax,ymin,ymax,zmin,zmax)
    tree.buildTree( minLevels=2, maxLevels=2, divideTolerance=0.001)
    
    
    

#     print(tree.root.children[0,1,1].children[1,1,0].gridpoints)
#     print()
#     cellFromListAccessedByAnotherCell = tree.root.children[0,1,1].children[1,1,0].tree.masterList[50][1]
# #     print(tree.root.children[0,1,1].children[1,1,0].tree.masterList[50])
#     print('cellFromListAccessedByAnotherCell', cellFromListAccessedByAnotherCell)
#     print("that cell's gridpoints ", cellFromListAccessedByAnotherCell.gridpoints)
    
#     tree.walkTree('uniqueID', storeOutput=False, leavesOnly=False)
#     tree.walkTree('neighbors', storeOutput=False, leavesOnly=False)
         
#     for element in tree.masterList:
#         print(element)
#     print(tree.masterList)
    
#     print(tree.root.children[0,1,1].tree.masterList[0])
# 
#     for element in tree.masterList:
#         if element[0] == '222212':
#             print(element)
        
#     tree.computePotentialOnTree(epsilon=0)
#     print('\nPotential Error:        %.3g mHartree' %float((-1.0-tree.totalPotential)*1000.0))
#          
#     tree.computeKineticOnTree()
#     print('Kinetic Error:           %.3g mHartree' %float((0.5-tree.totalKinetic)*1000.0))
#              
#     tree.visualizeMesh(attributeForColoring='psi')
#     print('Visualization Complete.')