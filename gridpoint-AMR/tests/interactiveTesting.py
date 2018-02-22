'''
Created on Feb 21, 2018

@author: nathanvaughn
'''
from dataStructs import Tree, Cell, GridPoint
import numpy as np


def setupForInteractiveTesting():
    testTree = Tree(-5,5,-5,5,-5,5)
    testTree.buildTree( minLevels=0, maxLevels=8, divideTolerance=0.1)