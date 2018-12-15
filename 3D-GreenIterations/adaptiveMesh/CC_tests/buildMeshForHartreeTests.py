'''
Created on Mar 5, 2018

@author: nathanvaughn
'''
import unittest
import sys
import numpy as np
import time
sys.path.append('../src/dataStructures')
sys.path.append('../src/utilities')
import itertools
ThreeByThreeByThree = [element for element in itertools.product(range(3),range(3),range(3))]
from numpy import pi, sqrt, exp
from scipy.special import erf


from TreeStruct_CC import Tree
from convolution import gpuPoissonConvolution, gpuPoissonConvolutionRegularized




def buildMesh(divideParameter):
    '''
    setUp() gets called before doing the tests below.
    '''
    inputFile ='../src/utilities/molecularConfigurations/dummyAtomAuxiliary.csv'
#         inputFile ='../src/utilities/molecularConfigurations/oxygenAtomAuxiliary.csv'
#         inputFile ='../src/utilities/molecularConfigurations/carbonMonoxideAuxiliary.csv'
#         inputFile ='../src/utilities/molecularConfigurations/carbonAtomAuxiliary.csv'
    xmin = ymin = zmin = -20
    xmax = ymax = zmax = 20
    order=5
    minDepth=3
    maxDepth=20
    divideCriterion='LW5'
    
    [coordinateFile, outputFile] = np.genfromtxt(inputFile,dtype="|U100")[:2]
    [nElectrons, nOrbitals, Eband, Ekinetic, Eexchange, Ecorrelation, Eelectrostatic, Etotal, gaugeShift] = np.genfromtxt(inputFile)[2:]
    nElectrons = int(nElectrons)
    nOrbitals = int(nOrbitals)
    
    print([coordinateFile, outputFile, nElectrons, nOrbitals, 
 Etotal, Eexchange, Ecorrelation, Eband, gaugeShift])
    tree = Tree(xmin,xmax,order,ymin,ymax,order,zmin,zmax,order,nElectrons,nOrbitals,maxDepthAtAtoms=maxDepth,gaugeShift=gaugeShift,
            coordinateFile=coordinateFile,inputFile=inputFile)#, iterationOutFile=outputFile)

    tree.buildTree( minLevels=minDepth, maxLevels=maxDepth, initializationType='atomic',divideCriterion=divideCriterion, divideParameter=divideParameter, printTreeProperties=True)
    tree.occupations = np.array([2])
    
    
    print()
    print()
    

            
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
#     unittest.main()
    buildMesh(divideParameter)