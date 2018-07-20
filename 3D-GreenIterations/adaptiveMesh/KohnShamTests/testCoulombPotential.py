<<<<<<< HEAD
'''
Created on Jul 9, 2018

@author: nathanvaughn
'''
import unittest
from numpy import sqrt,pi,exp
from scipy.special import erf
import numpy as np
import os
import csv
import sys
sys.path.append('../src/dataStructures')
sys.path.append('../src/utilities')

from TreeStruct_CC import Tree
from convolution import gpuPoissonConvolution, gpuPoissonConvolutionSmoothing, gpuPoissonConvolutionSingularitySubtract



def setUp():
    xmin = ymin = zmin = -domainSize
    xmax = ymax = zmax =  domainSize
    printTreeProperties=False
    coordinateFile = '../src/utilities/molecularConfigurations/hydrogenAtom.csv'
    tree = Tree(xmin,xmax,order,ymin,ymax,order,zmin,zmax,order,coordinateFile, printTreeProperties=printTreeProperties)

    tree.buildTree( minLevels, maxLevels, divideCriterion, divideParameter, printTreeProperties=printTreeProperties)

    alpha = 1
    for _,cell in tree.masterList:
        if cell.leaf==True:
            for i,j,k in cell.PxByPyByPz:
                gp = cell.gridpoints[i,j,k]
                r = sqrt(gp.x*gp.x + gp.y*gp.y + gp.z*gp.z)
                
                """ test case 1 """
#                     gp.rho = exp(-sqrt(4*pi)*r)/r
#                     gp.trueV = -exp(-sqrt(4*pi)*r)/r

                """ test case 2 """
#                 gp.rho = sqrt(36*pi)*alpha**2 / ( 4*pi*r**2 + alpha**2 )**(5/2)
#                 gp.trueV = sqrt(4*pi) / ( 4*pi*r**2 + alpha**2 )**(1/2)
                
                
                """ test case 3 """
                sigma = 0.05
                gp.rho = 1/(sigma**3*(2*pi)**(3/2))*exp(-r**2/(2*sigma**2))
                gp.trueV = (1/r)*erf(r/(sqrt(2)*sigma))

    
    print('Number of Cells:  ', tree.numberOfCells)
    print('Number of Points: ', tree.numberOfGridpoints)
    print('Maximum Depth:    ', tree.maxDepthAchieved)
    
    return tree
        




def testCoulombPotential(tree):
    
    def computeErrors(tree):
        L2err   = 0.0
        LInferr = 0.0
        
        x=-999
        y=-999
        z=-999
        rho = -999
        V = -999
        
#             xr=-999
#             yr=-999
#             zr=-999
#             rhor = -999
#             Vr = -999
        
        trueVintegral = 0.0
        for _,cell in tree.masterList:
            errors = np.zeros(np.shape(cell.gridpoints))
            trueVarray = np.zeros(np.shape(cell.gridpoints))
            if cell.leaf==True:
                for i,j,k in cell.PxByPyByPz:
                    gp = cell.gridpoints[i,j,k]
                    error = abs(gp.trueV - gp.v_coulomb)
                    errors[i,j,k] = error**2
                    trueVarray[i,j,k] = gp.trueV**2
#                     randCheck = np.random.randint(100000)
#                     if randCheck < 5:
#                         xr = gp.x
#                         yr = gp.y
#                         zr = gp.z
#                         rhor = gp.rho
#                         trueVr = gp.trueV
#                         Vr = gp.v_coulomb
#                         print('Random Check x,y,z:  ', xr,yr,zr)
#                         print('Random Check rho, trueVr, V_coulomb: ', rhor, trueVr, Vr)
#                         print()
                    if error > LInferr:
                        LInferr = error
                        x = gp.x
                        y = gp.y
                        z = gp.z
                        rho = gp.rho
                        trueV = gp.trueV
                        V = gp.v_coulomb
                    
                L2err += np.sum(cell.w * errors)
                trueVintegral += np.sum(cell.w * trueVarray)
        print('Location of LInferr:       ', x,y,z)
        print('Rho, trueV, computedV at LInferr: ', rho, trueV, V)
        L2err = sqrt(L2err)
        trueVintegral = sqrt(trueVintegral)
        print('Sqrt( int( trueV**2 ) ) = ', trueVintegral)
        L2err = L2err/trueVintegral
        return L2err, LInferr
    
    targets = tree.extractLeavesDensity()  # extract the target point locations.  Currently, these are all 27 gridpoints per cell (no redundancy)
    sources = tree.extractLeavesDensity()  # extract the source point locations.  Currently, these are just all the leaf midpoints

    threadsPerBlock = 512
    blocksPerGrid = (len(targets) + (threadsPerBlock - 1)) // threadsPerBlock  # compute the number of blocks based on N and threadsPerBlock


    V_coulombNew = np.zeros((len(targets)))
    if type==0:
        gpuPoissonConvolution[blocksPerGrid, threadsPerBlock](targets,sources,V_coulombNew)  # call the GPU convolution 
    elif type ==1:
        gpuPoissonConvolutionSmoothing[blocksPerGrid, threadsPerBlock](targets,sources,V_coulombNew,epsilon)
    elif type==2:
        gpuPoissonConvolutionSingularitySubtract[blocksPerGrid, threadsPerBlock](targets,sources,V_coulombNew,epsilon)  # call the GPU convolution 
    else:
        print('Type must be either 0 (skipping), 1 (smoothing), or 2 (subtracting)')
#     gpuPoissonConvolution[blocksPerGrid, threadsPerBlock](targets,sources,V_coulombNew)  # call the GPU convolution 
    tree.importVcoulombOnLeaves(V_coulombNew)

    L2err, LInferr = computeErrors(tree)
    
    print('L2 error:        ', L2err)
    print('LInf error:      ', LInferr)
    print()
    
    header = ['domainSize','minDepth','maxDepth','order','numberOfCells','numberOfPoints',
              'divideCriterion','divideParameter',
              'singularitySmoothed', 'singularitySubtracted','parameter', 
              'L2Err', 'LinfErr']
    
    myData = [domainSize,tree.minDepthAchieved,tree.maxDepthAchieved,tree.px,tree.numberOfCells,tree.numberOfGridpoints,
              divideCriterion,divideParameter,
              singularitySmoothed, singularitySubtracted, epsilon,
              L2err, LInferr]
    

    if not os.path.isfile(outFile):
        myFile = open(outFile, 'a')
        with myFile:
            writer = csv.writer(myFile)
            writer.writerow(header) 
            
    myFile = open(outFile, 'a')
    with myFile:
        writer = csv.writer(myFile)
        writer.writerow(myData)  

if __name__ == "__main__":
    domainSize          = float(sys.argv[1])
    minLevels           = int(sys.argv[2])
    maxLevels           = int(sys.argv[3])
    order               = int(sys.argv[4])
    type                = int(sys.argv[5])
    epsilon             = float(sys.argv[6])
    divideCriterion     = str(sys.argv[7])
    divideParameter     = float(sys.argv[8])
    outFile             = str(sys.argv[9])
    
    if type==1:
        singularitySmoothed = True
    else:
        singularitySmoothed = False
    if type==2:
        singularitySubtracted = True
    else:
        singularitySubtracted = False

    
    tree = setUp()
=======
'''
Created on Jul 9, 2018

@author: nathanvaughn
'''
import unittest
from numpy import sqrt,pi,exp
from scipy.special import erf
import numpy as np
import os
import csv
import sys
sys.path.append('../src/dataStructures')
sys.path.append('../src/utilities')

from TreeStruct_CC import Tree
from convolution import gpuPoissonConvolution, gpuPoissonConvolutionSmoothing, gpuPoissonConvolutionSingularitySubtract



def setUp():
    xmin = ymin = zmin = -domainSize
    xmax = ymax = zmax =  domainSize
    printTreeProperties=False
    coordinateFile = '../src/utilities/molecularConfigurations/hydrogenAtom.csv'
    tree = Tree(xmin,xmax,order,ymin,ymax,order,zmin,zmax,order,coordinateFile, printTreeProperties=printTreeProperties)

    tree.buildTree( minLevels, maxLevels, divideCriterion, divideParameter, printTreeProperties=printTreeProperties)

    alpha = 1
    for _,cell in tree.masterList:
        if cell.leaf==True:
            for i,j,k in cell.PxByPyByPz:
                gp = cell.gridpoints[i,j,k]
                r = sqrt(gp.x*gp.x + gp.y*gp.y + gp.z*gp.z)
                
                """ test case 1 """
#                     gp.rho = exp(-sqrt(4*pi)*r)/r
#                     gp.trueV = -exp(-sqrt(4*pi)*r)/r

                """ test case 2 """
#                 gp.rho = sqrt(36*pi)*alpha**2 / ( 4*pi*r**2 + alpha**2 )**(5/2)
#                 gp.trueV = sqrt(4*pi) / ( 4*pi*r**2 + alpha**2 )**(1/2)
                
                
                """ test case 3 """
                sigma = 0.05
                gp.rho = 1/(sigma**3*(2*pi)**(3/2))*exp(-r**2/(2*sigma**2))
                gp.trueV = (1/r)*erf(r/(sqrt(2)*sigma))

    
    print('Number of Cells:  ', tree.numberOfCells)
    print('Number of Points: ', tree.numberOfGridpoints)
    print('Maximum Depth:    ', tree.maxDepthAchieved)
    
    return tree
        




def testCoulombPotential(tree):
    
    def computeErrors(tree):
        L2err   = 0.0
        LInferr = 0.0
        
        x=-999
        y=-999
        z=-999
        rho = -999
        V = -999
        
#             xr=-999
#             yr=-999
#             zr=-999
#             rhor = -999
#             Vr = -999
        
        trueVintegral = 0.0
        for _,cell in tree.masterList:
            errors = np.zeros(np.shape(cell.gridpoints))
            trueVarray = np.zeros(np.shape(cell.gridpoints))
            if cell.leaf==True:
                for i,j,k in cell.PxByPyByPz:
                    gp = cell.gridpoints[i,j,k]
                    error = abs(gp.trueV - gp.v_coulomb)
                    errors[i,j,k] = error**2
                    trueVarray[i,j,k] = gp.trueV**2
#                     randCheck = np.random.randint(100000)
#                     if randCheck < 5:
#                         xr = gp.x
#                         yr = gp.y
#                         zr = gp.z
#                         rhor = gp.rho
#                         trueVr = gp.trueV
#                         Vr = gp.v_coulomb
#                         print('Random Check x,y,z:  ', xr,yr,zr)
#                         print('Random Check rho, trueVr, V_coulomb: ', rhor, trueVr, Vr)
#                         print()
                    if error > LInferr:
                        LInferr = error
                        x = gp.x
                        y = gp.y
                        z = gp.z
                        rho = gp.rho
                        trueV = gp.trueV
                        V = gp.v_coulomb
                    
                L2err += np.sum(cell.w * errors)
                trueVintegral += np.sum(cell.w * trueVarray)
        print('Location of LInferr:       ', x,y,z)
        print('Rho, trueV, computedV at LInferr: ', rho, trueV, V)
        L2err = sqrt(L2err)
        trueVintegral = sqrt(trueVintegral)
        print('Sqrt( int( trueV**2 ) ) = ', trueVintegral)
        L2err = L2err/trueVintegral
        return L2err, LInferr
    
    targets = tree.extractLeavesDensity()  # extract the target point locations.  Currently, these are all 27 gridpoints per cell (no redundancy)
    sources = tree.extractLeavesDensity()  # extract the source point locations.  Currently, these are just all the leaf midpoints

    threadsPerBlock = 512
    blocksPerGrid = (len(targets) + (threadsPerBlock - 1)) // threadsPerBlock  # compute the number of blocks based on N and threadsPerBlock


    V_coulombNew = np.zeros((len(targets)))
    if type==0:
        gpuPoissonConvolution[blocksPerGrid, threadsPerBlock](targets,sources,V_coulombNew)  # call the GPU convolution 
    elif type ==1:
        gpuPoissonConvolutionSmoothing[blocksPerGrid, threadsPerBlock](targets,sources,V_coulombNew,epsilon)
    elif type==2:
        gpuPoissonConvolutionSingularitySubtract[blocksPerGrid, threadsPerBlock](targets,sources,V_coulombNew,epsilon)  # call the GPU convolution 
    else:
        print('Type must be either 0 (skipping), 1 (smoothing), or 2 (subtracting)')
#     gpuPoissonConvolution[blocksPerGrid, threadsPerBlock](targets,sources,V_coulombNew)  # call the GPU convolution 
    tree.importVcoulombOnLeaves(V_coulombNew)

    L2err, LInferr = computeErrors(tree)
    
    print('L2 error:        ', L2err)
    print('LInf error:      ', LInferr)
    print()
    
    header = ['domainSize','minDepth','maxDepth','order','numberOfCells','numberOfPoints',
              'divideCriterion','divideParameter',
              'singularitySmoothed', 'singularitySubtracted','parameter', 
              'L2Err', 'LinfErr']
    
    myData = [domainSize,tree.minDepthAchieved,tree.maxDepthAchieved,tree.px,tree.numberOfCells,tree.numberOfGridpoints,
              divideCriterion,divideParameter,
              singularitySmoothed, singularitySubtracted, epsilon,
              L2err, LInferr]
    

    if not os.path.isfile(outFile):
        myFile = open(outFile, 'a')
        with myFile:
            writer = csv.writer(myFile)
            writer.writerow(header) 
            
    myFile = open(outFile, 'a')
    with myFile:
        writer = csv.writer(myFile)
        writer.writerow(myData)  

if __name__ == "__main__":
    domainSize          = float(sys.argv[1])
    minLevels           = int(sys.argv[2])
    maxLevels           = int(sys.argv[3])
    order               = int(sys.argv[4])
    type                = int(sys.argv[5])
    epsilon             = float(sys.argv[6])
    divideCriterion     = str(sys.argv[7])
    divideParameter     = float(sys.argv[8])
    outFile             = str(sys.argv[9])
    
    if type==1:
        singularitySmoothed = True
    else:
        singularitySmoothed = False
    if type==2:
        singularitySubtracted = True
    else:
        singularitySubtracted = False

    
    tree = setUp()
>>>>>>> refs/remotes/eclipse_auto/master
    testCoulombPotential(tree)