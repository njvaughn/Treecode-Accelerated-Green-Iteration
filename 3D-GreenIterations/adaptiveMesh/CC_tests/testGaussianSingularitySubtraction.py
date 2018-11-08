'''
Created on Mar 5, 2018

@author: nathanvaughn
'''
import unittest
import sys
import os
import csv
import numpy as np
import time
sys.path.append('../src/dataStructures')
sys.path.append('../src/utilities')
import itertools
from numpy import pi, sqrt, exp
from scipy.special import erf


from TreeStruct_CC import Tree
from convolution import gpuPoissonConvolution, gpuHartreeGaussianSingularitySubract, gpuHelmholtzConvolutionHybridSubractSingularity_gaussian_no_cusp
from convolution import gpuHelmholtzConvolution_skip_generic, gpuHelmholtzConvolution_subtract_generic, gpuHelmholtzConvolutionSubractSingularity_gaussian
from convolution_selfCell import gpuHelmholtzConvolution_skip_generic_selfCell, gpuHelmholtzConvolution_subtract_generic_selfCell
from meshUtilities import ChebLaplacian3D
from GaussianDensityTestCase import *
from HydrogenicDensityTestCase import *


domainSize                  = int(sys.argv[1])
minDepth                    = int(sys.argv[2])
maxDepth                    = int(sys.argv[3])
order                       = int(sys.argv[4])
subtractSingularity         = int(sys.argv[5])
smoothingN                  = int(sys.argv[6])
smoothingEps                = float(sys.argv[7])
divideCriterion             = str(sys.argv[8])
divideParameter             = float(sys.argv[9])
energyTolerance             = float(sys.argv[10])
scfTolerance                = float(sys.argv[11])
outputFile                  = str(sys.argv[12])
inputFile                   = str(sys.argv[13])
densityParameter            = float(sys.argv[14])
gaussianSubtractionAlpha    = float(sys.argv[15])
helmholtzShift              = float(sys.argv[16])
testCase                    = str(sys.argv[17])

vtkFileBase='/home/njvaughn/results_CO/orbitals'

def setUpTree():
    '''
    setUp() gets called before every test below.
    '''
    xmin = ymin = zmin = -domainSize
    xmax = ymax = zmax = domainSize

    [coordinateFile, DummyOutputFile] = np.genfromtxt(inputFile,dtype="|U100")[:2]
    [nElectrons, nOrbitals, Eband, Ekinetic, Eexchange, Ecorrelation, Eelectrostatic, Etotal, gaugeShift] = np.genfromtxt(inputFile)[2:]
    nElectrons = int(nElectrons)
    nOrbitals = int(nOrbitals)
    
    
    print([coordinateFile, outputFile, nElectrons, nOrbitals, 
     Etotal, Eexchange, Ecorrelation, Eband, gaugeShift])
    tree = Tree(xmin,xmax,order,ymin,ymax,order,zmin,zmax,order,nElectrons,nOrbitals,maxDepthAtAtoms=maxDepth,gaugeShift=gaugeShift,
                coordinateFile=coordinateFile,inputFile=inputFile)#, iterationOutFile=outputFile)

    
    
    print('max depth ', maxDepth)
    tree.buildTree( minLevels=minDepth, maxLevels=maxDepth, initializationType='atomic',divideCriterion=divideCriterion, divideParameter=divideParameter, printTreeProperties=True)
    tree.occupations = np.array([2])
    
    
    
    if testCase == 'Gaussian':
        print('Setting density and true Hartree to the Gaussian test case functions, helmholtzShift =',helmholtzShift)
        setDensityToGaussian(tree,densityParameter)
        setTrueHartree(tree,densityParameter)
        setIntegrand(tree,helmholtzShift)
        tree.trueHartreeEnergy = hartreeEnergy(densityParameter)
        targets = tree.extractConvolutionIntegrand() 
        r = np.sqrt(targets[:,0]**2 + targets[:,1]**2 + targets[:,2]**2)
        tree.V_HartreeTrue = gaussianHartree(r,densityParameter)

    elif testCase == 'Hydrogenic':
        print('Setting density and true Hartree to the Hydrogenic test case functions, helmholtzShift =',helmholtzShift)
        setDensityToHydrogenic(tree,densityParameter)
        setTrueHydrogenicHartree(tree,densityParameter)
        setIntegrand(tree,helmholtzShift)
        tree.trueHartreeEnergy = hydrogenicHartreeEnergy(densityParameter)
        targets = tree.extractConvolutionIntegrand() 
        r = np.sqrt(targets[:,0]**2 + targets[:,1]**2 + targets[:,2]**2)
        tree.V_HartreeTrue = hydrogenicHartreePotential(r,densityParameter)
    else:
        print('Invalid test case option: choose Hydrogenic or Gaussian.')
        return
#     tree.computeSelfCellInterations(helmholtzShift)
    print()
    print()
    
    return tree


def HartreeCalculation(tree):

    print()
    targets = tree.extractConvolutionIntegrand() 
    sources = targets
    weights = np.copy(targets[:,4])   
    
#     targets_selfCell =  tree.extractConvolutionIntegrand_selfCell()
#     sources_selfCell =  targets_selfCell
    
    threadsPerBlock = 512
    blocksPerGrid = (tree.numberOfGridpoints + (threadsPerBlock - 1)) // threadsPerBlock  # compute the number of blocks based on N and threadsPerBlock


    V_HartreeNew = np.zeros((len(targets)))
    alphasq = gaussianSubtractionAlpha*gaussianSubtractionAlpha
    if helmholtzShift==0:
        print('Using Gaussian singularity subtraction, alpha = ', gaussianSubtractionAlpha)
#         gpuPoissonConvolution[blocksPerGrid, threadsPerBlock](targets,sources,V_HartreeNew)  # call the GPU convolution
        gpuHartreeGaussianSingularitySubract[blocksPerGrid, threadsPerBlock](targets,sources,V_HartreeNew, alphasq)  # call the GPU convolution
    else:
        print('helmholtzShift=',helmholtzShift,', using a Helmholtz convolution. ')
#         gpuHelmholtzConvolution_skip_generic[blocksPerGrid, threadsPerBlock](targets,sources,V_HartreeNew, helmholtzShift)  # call the GPU convolution
#         gpuHelmholtzConvolution_subtract_generic[blocksPerGrid, threadsPerBlock](targets,sources,V_HartreeNew, helmholtzShift)  # call the GPU convolution
        gpuHelmholtzConvolutionSubractSingularity_gaussian[blocksPerGrid, threadsPerBlock](targets,sources,V_HartreeNew, helmholtzShift, alphasq)  # call the GPU convolution
#         gpuHelmholtzConvolutionHybridSubractSingularity_gaussian_no_cusp[blocksPerGrid, threadsPerBlock](targets,sources,V_HartreeNew, helmholtzShift, alphasq)  # call the GPU convolution
         
#         gpuHelmholtzConvolution_skip_generic[blocksPerGrid, threadsPerBlock](targets,sources,V_HartreeNew, helmholtzShift)  # call the GPU convolution
#         gpuHelmholtzConvolution_skip_generic_selfCell[blocksPerGrid, threadsPerBlock](targets_selfCell,sources_selfCell,V_HartreeNew, helmholtzShift)  # call the GPU convolution
#         gpuHelmholtzConvolution_subtract_generic_selfCell[blocksPerGrid, threadsPerBlock](targets_selfCell,sources_selfCell,V_HartreeNew, helmholtzShift)  # call the GPU convolution
         
 
    tree.importVcoulombOnLeaves(V_HartreeNew)
    tree.updateVxcAndVeffAtQuadpoints()
     
    computedFromNumericalPotential = computeHartreeEnergyFromNumericalPotential(tree)
    computedFromAnalyticPotential = computeHartreeEnergyFromAnalyticPotential(tree)
    
    
    
     
    # Compute relative L2 error and Linf error
    L2Error = np.sqrt( np.sum( (V_HartreeNew - tree.V_HartreeTrue)**2 * weights  )  ) / np.sqrt( np.sum( (tree.V_HartreeTrue)**2 * weights  )  )
    LinfError = np.max(   np.abs( V_HartreeNew - tree.V_HartreeTrue )/np.abs( tree.V_HartreeTrue )  ) 
    idx = np.argmax(   np.abs( V_HartreeNew - tree.V_HartreeTrue )/np.abs( tree.V_HartreeTrue )  ) 
    
    
    print('True Hartree Energy:                       ', tree.trueHartreeEnergy)
    print('Computed from Analytic Potential:          ', computedFromAnalyticPotential)
    print('Computed from Numerical Potential:         ', computedFromNumericalPotential)
    print()
    print('Error for Analytic Potential:         %1.3e' %(computedFromAnalyticPotential - tree.trueHartreeEnergy))
    print('Error for Computed Potential:         %1.3e' %(computedFromNumericalPotential - tree.trueHartreeEnergy))
    print()
    print('L2 Error:                             %1.3e' %L2Error)
    print('Linf Error:                           %1.3e' %LinfError)
    print('Located at:                           ', targets[idx][0:3])
    print('Analytic value:                       ', tree.V_HartreeTrue[idx])
    print('Computed value:                       ', V_HartreeNew[idx])
    print()
    print()
    
    
    idx = np.argmax( tree.V_HartreeTrue )
    print('Max value of true VH is     ', tree.V_HartreeTrue[idx],' located at ', targets[idx][0:3])
    idx = np.argmax( V_HartreeNew )
    print('Max value of computed VH is ', V_HartreeNew[idx],' located at ', targets[idx][0:3])
    idx = np.argmin( tree.V_HartreeTrue )
    print('Min value of true VH is     ', tree.V_HartreeTrue[idx],' located at ', targets[idx][0:3])
    idx = np.argmin( V_HartreeNew )
    print('Min value of computed VH is ', V_HartreeNew[idx],' located at ', targets[idx][0:3])
#     
#     header = ['domainSize','minDepth','maxDepth','order','numberOfCells','numberOfPoints',
#               'divideCriterion','divideParameter','GaussianAlpha',
#               'trueHartreeEnergy', 'EnergyErrorFromAnalytic', 'EnergyErrorFromNumerical',
#               'L2Error','LinfError']
#     
#     myData = [domainSize,tree.minDepthAchieved,tree.maxDepthAchieved,tree.px,tree.numberOfCells,tree.numberOfGridpoints,
#               divideCriterion,divideParameter,gaussianSubtractionAlpha,
#               tree.trueHartreeEnergy, abs(computedFromAnalyticPotential-tree.trueHartreeEnergy), abs(computedFromNumericalPotential-tree.trueHartreeEnergy),
#               L2Error, LinfError]
#               
# 
#     
#     if not os.path.isfile(outputFile):
#         myFile = open(outputFile, 'a')
#         with myFile:
#             writer = csv.writer(myFile)
#             writer.writerow(header) 
#         
#         
#     
#     myFile = open(outputFile, 'a')
#     with myFile:
#         writer = csv.writer(myFile)
#         writer.writerow(myData)  
        
       
        
        
      
if __name__ == "__main__":

    print('='*70)
    print('='*70)
    print('='*70,'\n')
    tree = setUpTree()
    HartreeCalculation(tree)
    
    
    
    