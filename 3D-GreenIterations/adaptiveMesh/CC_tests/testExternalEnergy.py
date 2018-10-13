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

def exponentialDensity(r,alpha):
        return exp(-alpha * r)

def externalPotential(r,Z):
    return -Z/r
    

def externalEnergy(alpha,Z):
    return -4*pi*Z/alpha**2



def setDensityToExponential(tree,alpha):
    
    for _,cell in tree.masterList:
        if cell.leaf == True:
            # set density on the primary mesh
            for i,j,k in cell.PxByPyByPz:
                gp = cell.gridpoints[i,j,k]
                r = sqrt( gp.x**2 + gp.y**2 + gp.z**2 )
                gp.rho = exponentialDensity(r,alpha)
            
            # set density on the secondary mesh 
            for i,j,k in cell.PxByPyByPz_density:
                dp = cell.densityPoints[i,j,k]
                r = sqrt( dp.x**2 + dp.y**2 + dp.z**2 )
                dp.rho = exponentialDensity(r,alpha)
                
def setExternalPotential(tree,Z):
    
    for _,cell in tree.masterList:
        if cell.leaf == True:
            for i,j,k in cell.PxByPyByPz:
                gp = cell.gridpoints[i,j,k]
                r = sqrt( gp.x**2 + gp.y**2 + gp.z**2 )
                gp.v_ext = externalPotential(r,Z)


                
def integrateCellDensityAgainst__(cell,integrand):
            rho = np.empty((cell.px,cell.py,cell.pz))
            pot = np.empty((cell.px,cell.py,cell.pz))
            
            for i,j,k in cell.PxByPyByPz:
                gp = cell.gridpoints[i,j,k]
                rho[i,j,k] = gp.rho
                pot[i,j,k] = getattr(gp,integrand)
            
            return np.sum( cell.w * rho * pot)
        

def computeExternalEnergy(tree):
    E = 0.0
    for _,cell in tree.masterList:
        if cell.leaf == True:
            E += integrateCellDensityAgainst__(cell,'v_ext') 
    return E
    

class TestEnergyComputation(unittest.TestCase):

    @classmethod
    def setUpClass(self):
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
        divideParameter=3000
        self.alpha = 2
        self.Z = 8
        
        [coordinateFile, outputFile] = np.genfromtxt(inputFile,dtype="|U100")[:2]
        [nElectrons, nOrbitals, Eband, Ekinetic, Eexchange, Ecorrelation, Eelectrostatic, Etotal, gaugeShift] = np.genfromtxt(inputFile)[2:]
        nElectrons = int(nElectrons)
        nOrbitals = int(nOrbitals)
        
        print([coordinateFile, outputFile, nElectrons, nOrbitals, 
     Etotal, Eexchange, Ecorrelation, Eband, gaugeShift])
        self.tree = Tree(xmin,xmax,order,ymin,ymax,order,zmin,zmax,order,nElectrons,nOrbitals,maxDepthAtAtoms=maxDepth,gaugeShift=gaugeShift,
                coordinateFile=coordinateFile,inputFile=inputFile)#, iterationOutFile=outputFile)
    
        self.tree.buildTree( minLevels=minDepth, maxLevels=maxDepth, initializationType='atomic',divideCriterion=divideCriterion, divideParameter=divideParameter, printTreeProperties=True)
#         self.tree.occupations = np.array([2,2,4/3,4/3,4/3])
        self.tree.occupations = np.array([2,2])
        
        setExternalPotential(self.tree,self.Z)
        setDensityToExponential(self.tree,self.alpha)
        print()
        print()
    


    def testEnergyWithAnalyticPotential(self):
        
        computedExternalEnergy = computeExternalEnergy(self.tree)
        trueExternalEnergy = externalEnergy(self.alpha,self.Z)
        print('True External Energy:      ',trueExternalEnergy)
        print('Computed External Energy:  ',computedExternalEnergy)
        print('Error:                      %1.3e' %(computedExternalEnergy-trueExternalEnergy))
        print()
        self.assertAlmostEqual(trueExternalEnergy, computedExternalEnergy, 8, 
                               "Analytic and Computed Energy not agreeing well enough")


            
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()