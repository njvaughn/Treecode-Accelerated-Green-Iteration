'''
The main Tree data structure.  The root of the tree is a Cell object that is comprised of the 
entire domain.  The tree gets built by dividing the root cell, recursively, based on the set 
divideInto8 condition.  The current implementation uses the variation of phi within a cell to 
dictate whether or not it divides.  

Cells can perform recursive functions on the tree.  The tree can also extract all gridpoints or
all midpoints as arrays which can be fed in to the GPU kernels, or other tree-external functions.
-- 03/20/2018 NV

@author: nathanvaughn
'''

import numpy as np
import pylibxc
import itertools
import os
import csv
import vtk
import matplotlib
# matplotlib.use('TkAgg')
matplotlib.use('Agg')  # to not display
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from GridpointStruct import GridPoint
from CellStruct_CC import Cell
from AtomStruct import Atom
from hydrogenAtom import potential, trueWavefunction, trueEnergy
from meshUtilities import *
from timer import Timer

ThreeByThreeByThree = [element for element in itertools.product(range(3),range(3),range(3))]
ThreeByThree = [element for element in itertools.product(range(3),range(3))]
TwoByTwoByTwo = [element for element in itertools.product(range(2),range(2),range(2))]
FiveByFiveByFive = [element for element in itertools.product(range(5),range(5),range(5))]

class Tree(object):
    '''
    Tree object. Constructed of cells, which are composed of gridpoint objects.  
    Trees contain their root, as well as their masterList.
    '''
        
    """
    INTIALIZATION FUNCTIONS
    """
    def __init__(self, xmin,xmax,px,ymin,ymax,py,zmin,zmax,pz,coordinateFile,xcFunctional="LDA_XC_LP_A",polarization="unpolarized"):
        '''
        Tree constructor:  
        First construct the gridpoints for cell consisting of entire domain.  
        Then construct the cell that are composed of gridpoints. 
        Then construct the root of the tree.
        '''
        self.xmin = xmin
        self.xmax = xmax
        self.px = px
        self.ymin = ymin
        self.ymax = ymax
        self.py = py
        self.zmin = zmin
        self.zmax = zmax
        self.pz = pz
        self.PxByPyByPz = [element for element in itertools.product(range(self.px),range(self.py),range(self.pz))]
        
        self.XCfunc = pylibxc.LibXCFunctional(xcFunctional, polarization)
        
        # generate gridpoint objects.  
        xvec = ChebyshevPoints(self.xmin,self.xmax,self.px)
        yvec = ChebyshevPoints(self.ymin,self.ymax,self.py)
        zvec = ChebyshevPoints(self.zmin,self.zmax,self.pz)
        gridpoints = np.empty((px,py,pz),dtype=object)

        for i, j, k in self.PxByPyByPz:
            gridpoints[i,j,k] = GridPoint(xvec[i],yvec[j],zvec[k])
        
        # generate root cell from the gridpoint objects  
        self.root = Cell( self.xmin, self.xmax, self.px, 
                          self.ymin, self.ymax, self.py, 
                          self.zmin, self.zmax, self.pz, 
                          gridpoints, self )
        self.root.level = 0
        self.root.uniqueID = ''
        self.masterList = [[self.root.uniqueID, self.root]]
        
        self.initialDivideBasedOnNuclei(coordinateFile)
        print('\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('~~~~~~~~~~~~~~~~~~~~~~~ Atoms ~~~~~~~~~~~~~~~~~~~~~~~')
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        for i in range(len(self.atoms)):
            print('Z = %i located at (x, y, z) = (%6.3f, %6.3f, %6.3f)' 
                  %(self.atoms[i].atomicNumber, self.atoms[i].x,self.atoms[i].y,self.atoms[i].z))
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')
        
    def initialDivideBasedOnNuclei(self, coordinateFile):
            
        
        def recursiveDivideByAtom(self,Atom,Cell):
            # Atom is in this cell.  Check if this cell has children.  If so, find the child that contains
            # the atom.  If not, divideInto8 the cell.
            if hasattr(Cell, "children"):
#                 x = Cell.children[0,0,0].xmax
#                 y = Cell.children[0,0,0].ymax
#                 z = Cell.children[0,0,0].zmax
                (ii,jj,kk) = np.shape(Cell.children)
                for i in range(ii):
                    for j in range(jj):
                        for k in range(kk):
#                 for i,j,k in TwoByTwoByTwo: # this should catch cases where atom is on the boundary of a previous cut
                            if ( (Atom.x <= Cell.children[i,j,k].xmax) and (Atom.x >= Cell.children[i,j,k].xmin) ):
                                if ( (Atom.y <= Cell.children[i,j,k].ymax) and (Atom.y >= Cell.children[i,j,k].ymin) ):
                                    if ( (Atom.z <= Cell.children[i,j,k].zmax) and (Atom.z >= Cell.children[i,j,k].zmin) ):
                                        recursiveDivideByAtom(self, Atom, Cell.children[i,j,k])


  
            else:  # sets the divideInto8 location.  If atom is on the boundary, sets divideInto8 location to None for that dimension
                xdiv = Atom.x
                ydiv = Atom.y
                zdiv = Atom.z
                if ( (Atom.x == Cell.xmax) or (Atom.x == Cell.xmin) ):
                    xdiv = None
                if ( (Atom.y == Cell.ymax) or (Atom.y == Cell.ymin) ):
                    ydiv = None
                if ( (Atom.z == Cell.zmax) or (Atom.z == Cell.zmin) ):
                    zdiv = None
                    
                Cell.divide(xdiv, ydiv, zdiv)

        atomData = np.genfromtxt(coordinateFile,delimiter=',',dtype=None)
#         print(np.shape(atomData))
#         print(len(atomData))
        if np.shape(atomData)==(4,):
            self.atoms = np.empty((1,),dtype=object)
            atom = Atom(atomData[0],atomData[1],atomData[2],atomData[3])
            self.atoms[0] = atom
        else:
            self.atoms = np.empty((len(atomData),),dtype=object)
            for i in range(len(atomData)):
                atom = Atom(atomData[i,0],atomData[i,1],atomData[i,2],atomData[i,3])
                self.atoms[i] = atom
                self.atoms[i] = atom
        
        for atom in self.atoms:
            recursiveDivideByAtom(self,atom,self.root)
        
#         self.exportMeshVTK('/Users/nathanvaughn/Desktop/aspectRatioBefore2.vtk')
        for cell in self.masterList:
            if cell[1].leaf==True:
                cell[1].divideIfAspectRatioExceeds(1.5)
      
    def buildTree(self,minLevels,maxLevels, divideCriterion, divideParameter, printNumberOfCells=False, printTreeProperties = True): # call the recursive divison on the root of the tree
        # max depth returns the maximum depth of the tree.  maxLevels is the limit on how large the tree is allowed to be,
        # regardless of division criteria
        # N is roughly the number of grid points.  It is used to generate the density function.
        timer = Timer()
        def recursiveDivide(self, Cell, minLevels, maxLevels, divideCriterion, divideParameter, levelCounter, printNumberOfCells, maxDepthAchieved=0, minDepthAchieved=100, currentLevel=0):
            levelCounter += 1
            
            if hasattr(Cell, "children"):
#                 print('Cell already has children')
                (ii,jj,kk) = np.shape(Cell.children)

                for i in range(ii):
                    for j in range(jj):
                        for k in range(kk):
#                     print('Calling recursive divideInto8 on child:')
#                     print('xmin, xmax: ',Cell.children[i,j,k].xmin,Cell.children[i,j,k].xmax)
#                     print('ymin, ymax: ',Cell.children[i,j,k].ymin,Cell.children[i,j,k].ymax)
#                     print('zmin, zmax: ',Cell.children[i,j,k].zmin,Cell.children[i,j,k].zmax)
                            maxDepthAchieved, minDepthAchieved, levelCounter = recursiveDivide(self,Cell.children[i,j,k], 
                                                                                minLevels, maxLevels, divideCriterion, divideParameter, 
                                                                                levelCounter, printNumberOfCells, maxDepthAchieved, 
                                                                                minDepthAchieved, currentLevel+1)
            
            elif currentLevel < maxLevels:
                
                if currentLevel < minLevels:
                    Cell.divideFlag = True 
#                     print('dividing cell ', Cell.uniqueID, ' because it is below the minimum level')
                else:  
                    if (divideCriterion == 'LW1') or (divideCriterion == 'LW2') or (divideCriterion == 'LW3'):
#                         print('checking divide criterion for cell ', Cell.uniqueID)
                        Cell.checkIfAboveMeshDensity(divideParameter,divideCriterion)  
                    else:                        
                        Cell.checkIfCellShouldDivide(divideParameter)
                    
                if Cell.divideFlag == True:
                    xdiv = (Cell.xmax + Cell.xmin)/2   
                    ydiv = (Cell.ymax + Cell.ymin)/2   
                    zdiv = (Cell.zmax + Cell.zmin)/2   
                    Cell.divideInto8(xdiv, ydiv, zdiv, printNumberOfCells)
#                     for i,j,k in TwoByTwoByTwo: # update the list of cells
#                         self.masterList.append([CellStruct.children[i,j,k].uniqueID, CellStruct.children[i,j,k]])
                    for i,j,k in TwoByTwoByTwo:
                        maxDepthAchieved, minDepthAchieved, levelCounter = recursiveDivide(self,Cell.children[i,j,k], minLevels, maxLevels, divideCriterion, divideParameter, levelCounter, printNumberOfCells, maxDepthAchieved, minDepthAchieved, currentLevel+1)
                else:
                    minDepthAchieved = min(minDepthAchieved, currentLevel)
                    
                    
            maxDepthAchieved = max(maxDepthAchieved, currentLevel)                                                                                                                                                       
            return maxDepthAchieved, minDepthAchieved, levelCounter
        
        timer.start()
        levelCounter=0
        self.maxDepthAchieved, self.minDepthAchieved, self.treeSize = recursiveDivide(self, self.root, minLevels, maxLevels, divideCriterion, divideParameter, levelCounter, printNumberOfCells, maxDepthAchieved=0, minDepthAchieved=maxLevels, currentLevel=0 )
        timer.stop()
        
        """ Count the number of unique leaf cells and gridpoints """
        self.numberOfGridpoints = 0
        self.numberOfCells = 0
        for element in self.masterList:
            if element[1].leaf==True:
                self.numberOfCells += 1
                for i,j,k in self.PxByPyByPz:
                    if not hasattr(element[1].gridpoints[i,j,k], "counted"):
                        self.numberOfGridpoints += 1
                        element[1].gridpoints[i,j,k].counted = True
        
                        
        for element in self.masterList:
            for i,j,k in self.PxByPyByPz:
                if hasattr(element[1].gridpoints[i,j,k], "counted"):
                    element[1].gridpoints[i,j,k].counted = None
                    
        if printTreeProperties == True: 
            print("Tree build completed. \n"
                  "Domain Size:                 [%.1f, %.1f] \n"
                  "Divide Ciretion:             %s \n"
                  "Divide Parameter:            %1.2e \n"
                  "Total Number of Cells:       %i \n"
                  "Total Number of Leaf Cells:  %i \n"
                  "Total Number of Gridpoints:  %i \n"
                  "Minimum Depth                %i levels \n"
                  "Maximum Depth:               %i levels \n"
                  "Cell Order:                  %i \n"
                  "Construction time:           %.3g seconds."
                   
                  %(self.xmin, self.xmax, divideCriterion,divideParameter, self.treeSize, self.numberOfCells, self.numberOfGridpoints, self.minDepthAchieved,self.maxDepthAchieved, self.px, timer.elapsedTime))
                  
    def buildTreeOneCondition(self,minLevels,maxLevels,divideTolerance,printNumberOfCells=False, printTreeProperties = True): # call the recursive divison on the root of the tree
        # max depth returns the maximum depth of the tree.  maxLevels is the limit on how large the tree is allowed to be,
        # regardless of division criteria
        timer = Timer()
        def recursiveDivide(self, Cell, minLevels, maxLevels, divideTolerance, levelCounter, printNumberOfCells, maxDepthAchieved=0, minDepthAchieved=100, currentLevel=0):
            levelCounter += 1
            if currentLevel < maxLevels:
                
                if currentLevel < minLevels:
                    Cell.divideFlag = True 
                else:                             
                    Cell.checkIfCellShouldDivide(divideTolerance)
                    
                if Cell.divideFlag == True:   
                    Cell.divideInto8(printNumberOfCells)
#                     for i,j,k in TwoByTwoByTwo: # update the list of cells
#                         self.masterList.append([CellStruct.children[i,j,k].uniqueID, CellStruct.children[i,j,k]])
                    for i,j,k in TwoByTwoByTwo:
                        maxDepthAchieved, minDepthAchieved, levelCounter = recursiveDivide(self,Cell.children[i,j,k], minLevels, maxLevels, divideTolerance, levelCounter, printNumberOfCells, maxDepthAchieved, minDepthAchieved, currentLevel+1)
                else:
                    minDepthAchieved = min(minDepthAchieved, currentLevel)
                    
                    
            maxDepthAchieved = max(maxDepthAchieved, currentLevel)                                                                                                                                                       
            return maxDepthAchieved, minDepthAchieved, levelCounter
        
        timer.start()
        levelCounter=0
        self.maxDepthAchieved, self.minDepthAchieved, self.treeSize = recursiveDivide(self, self.root, minLevels, maxLevels, divideTolerance, levelCounter, printNumberOfCells, maxDepthAchieved=0, minDepthAchieved=maxLevels, currentLevel=0 )
        timer.stop()
        
        """ Count the number of unique gridpoints """
        self.numberOfGridpoints = 0
        for element in self.masterList:
            if element[1].leaf==True:
                for i,j,k in self.PxByPyByPz:
                    if not hasattr(element[1].gridpoints[i,j,k], "counted"):
                        self.numberOfGridpoints += 1
                        element[1].gridpoints[i,j,k].counted = True
        
                        
        for element in self.masterList:
            for i,j,k in self.PxByPyByPz:
                if hasattr(element[1].gridpoints[i,j,k], "counted"):
                    element[1].gridpoints[i,j,k].counted = None
                    
        if printTreeProperties == True: 
            print("Tree build completed. \n"
                  "Domain Size:                 [%.1f, %.1f] \n"
                  "Tolerance:                   %1.2e \n"
                  "Total Number of Cells:       %i \n"
                  "Total Number of Gridpoints:  %i \n"
                  "Minimum Depth                %i levels \n"
                  "Maximum Depth:               %i levels \n"
                  "Construction time:           %.3g seconds." 
                  %(self.xmin, self.xmax, divideTolerance, self.treeSize, self.numberOfGridpoints, self.minDepthAchieved,self.maxDepthAchieved,timer.elapsedTime))
            
    def buildTreeTwoConditions(self,minLevels,maxLevels, maxDx, divideTolerance1, divideTolerance2,printNumberOfCells=False, printTreeProperties = True): # call the recursive divison on the root of the tree
        # max depth returns the maximum depth of the tree.  maxLevels is the limit on how large the tree is allowed to be,
        # regardless of division criteria
        timer = Timer()
        def recursiveDivide(self, Cell, minLevels, maxLevels, maxDx, divideTolerance1, divideTolerance2, levelCounter, printNumberOfCells, maxDepthAchieved=0, minDepthAchieved=100, currentLevel=0):
            levelCounter += 1
            if currentLevel < maxLevels:
                
                if ( (currentLevel < minLevels) or (Cell.dx > maxDx)):
                    Cell.divideFlag = True 
                    
                else:                             
                    Cell.checkIfCellShouldDivideTwoConditions(divideTolerance1, divideTolerance2)
                    
                if Cell.divideFlag == True:   
                    Cell.divideInto8(printNumberOfCells)
#                     for i,j,k in TwoByTwoByTwo: # update the list of cells
#                         self.masterList.append([CellStruct.children[i,j,k].uniqueID, CellStruct.children[i,j,k]])
                    for i,j,k in TwoByTwoByTwo:
                        maxDepthAchieved, minDepthAchieved, levelCounter = recursiveDivide(self,Cell.children[i,j,k], minLevels, maxLevels, maxDx, divideTolerance1, divideTolerance2, levelCounter, printNumberOfCells, maxDepthAchieved, minDepthAchieved, currentLevel+1)
                else:
                    minDepthAchieved = min(minDepthAchieved, currentLevel)
                    
                    
            maxDepthAchieved = max(maxDepthAchieved, currentLevel)                                                                                                                                                       
            return maxDepthAchieved, minDepthAchieved, levelCounter
        
        timer.start()
        levelCounter=0
        self.maxDepthAchieved, self.minDepthAchieved, self.treeSize = recursiveDivide(self, self.root, minLevels, maxLevels, maxDx, divideTolerance1, divideTolerance2, levelCounter, printNumberOfCells, maxDepthAchieved=0, minDepthAchieved=maxLevels, currentLevel=0 )
        timer.stop()
        
        """ Count the number of unique gridpoints """
        self.numberOfGridpoints = 0
        for element in self.masterList:
            if element[1].leaf==True:
                for i,j,k in self.PxByPyByPz:
                    if not hasattr(element[1].gridpoints[i,j,k], "counted"):
                        self.numberOfGridpoints += 1
                        element[1].gridpoints[i,j,k].counted = True
        
                        
        for element in self.masterList:
            for i,j,k in self.PxByPyByPz:
                if hasattr(element[1].gridpoints[i,j,k], "counted"):
                    element[1].gridpoints[i,j,k].counted = None
                    
        if printTreeProperties == True: 
            print("Tree build completed. \n"
                  "Domain Size:                 [%.1f, %.1f] \n"
                  "Tolerance1:                  %1.2e \n"
                  "Tolerance2:                  %1.2e \n"
                  "Total Number of Cells:       %i \n"
                  "Total Number of Gridpoints:  %i \n"
                  "Minimum Depth                %i levels \n"
                  "Maximum Depth:               %i levels \n"
                  "Construction time:           %.3g seconds." 
                  %(self.xmin, self.xmax, divideTolerance1, divideTolerance2, self.treeSize, self.numberOfGridpoints, self.minDepthAchieved,self.maxDepthAchieved,timer.elapsedTime))
    
    def populatePhiWithAnalytic(self,n):
        for element in self.masterList:
            for i,j,k in self.PxByPyByPz:
                element[1].gridpoints[i,j,k].setAnalyticPhi(n)
        self.normalizeWavefunction()    

    
    """
    UPDATE  DENSITY AND EFFECTIVE POTENTIAL AT GRIDPOINTS
    """
    def updateCoulombPotentialAtQuadpoints(self):
        # call the convolution 
        pass
    
    def updateVxcAndVeffAtQuadpoints(self):
        
        def CellupdateVxcAndVeff(Cell,xcFunc):
            '''
            After density is updated the convolution gets called to update V_coulom.
            Now I need to update V_xc, then get the new value of V_eff. 
            '''
            
            rho = np.empty((Cell.px,Cell.py,Cell.pz))
            
            for i,j,k in Cell.PxByPyByPz:
                rho[i,j,k] = Cell.gridpoints[i,j,k].rho
                
            xcOutput = xcFunc.compute(rho)
            EXC = np.reshape(xcOutput['zk'],np.shape(rho))
            VRHO = np.reshape(xcOutput['vrho'],np.shape(rho))
            
            for i,j,k in self.PxByPyByPz:
                Cell.gridpoints[i,j,k].epsilon_xc = EXC[i,j,k]
                Cell.gridpoints[i,j,k].V_xc = VRHO[i,j,k]
                Cell.gridpoints[i,j,k].updateVeff()
            
        for element in self.masterList:
            Cell = element[1]
            if Cell.leaf == True:
                CellupdateVxcAndVeff(Cell,self.xcFunc)

    def updateDensityAtQuadpoints(self):
        def CellUpdateDensity(Cell):
            for i,j,k in self.PxByPyByPz:
                # for m in range(orbitals)
                Cell.gridpoints[i,j,k].rho = Cell.gridpoints[i,j,k].phi**2
        
        for element in self.masterList:
            Cell = element[1]
            if Cell.leaf == True:
                CellUpdateDensity(Cell)
            
    
    """
    ENERGY COMPUTATION FUNCTIONS
    """       
    def computeExternalPotential(self, epsilon=0, timePotential = False): 
       
        timer = Timer() 
 
        self.totalPotential = 0
        
        timer.start() 
        for element in self.masterList:
            Cell = element[1]
            if Cell.leaf == True:
                Cell.computePotential(epsilon)
                self.totalPotential += Cell.PE
                       
        timer.stop() 
        if timePotential == True:
            self.PotentialTime = timer.elapsedTime        

    def computeKinetic(self, timeKinetic = False):

        
        self.totalKinetic = 0
        timer = Timer()
        timer.start()
        for element in self.masterList:
            Cell = element[1]
            if Cell.leaf == True:
                Cell.computeKinetic()
                self.totalKinetic += Cell.KE
        timer.stop()
        if timeKinetic == True:
            self.KineticTime = timer.elapsedTime
            
    def updateEnergy(self,epsilon=0.0):
        self.computeKinetic()
        self.computeExternalPotential(epsilon)
        self.E = self.totalKinetic + self.totalPotential
            
                    
    """
    NORMALIZATION, ORTHOGONALIZATION, AND WAVEFUNCTION ERRORS
    """      
    def computeWaveErrors(self,energyLevel,normalizationFactor):
        
        # need normalizationFactor because the analytic wavefunctions aren't normalized for this finite domain.
        maxErr = 0.0
        errorsIfSameSign = []
        errorsIfDifferentSign = []
        phiComputed = np.zeros((self.px,self.py,self.pz))
        phiAnalytic = np.zeros((self.px,self.py,self.pz))
        for element in self.masterList:
            if element[1].leaf == True:
                for i,j,k in self.PxByPyByPz:
                    gridpt = element[1].gridpoints[i,j,k]
                    phiComputed[i,j,k] = gridpt.phi
                    phiAnalytic[i,j,k] = normalizationFactor*trueWavefunction(energyLevel,gridpt.x,gridpt.y,gridpt.z)
                errorsIfSameSign.append( np.sum( (phiAnalytic-phiComputed)**2*element[1].w ))
                errorsIfDifferentSign.append( np.sum( (phiAnalytic+phiComputed)**2*element[1].w ))
#                 errorsIfSameSign.append( (midpoint.phi - normalizationFactor*trueWavefunction(energyLevel,midpoint.x,midpoint.y,midpoint.z))**2 * element[1].volume)
#                 errorsIfDifferentSign.append( (midpoint.phi + normalizationFactor*trueWavefunction(energyLevel,midpoint.x,midpoint.y,midpoint.z))**2 * element[1].volume)
                absErr = np.max(np.abs( phiAnalytic-phiComputed ))
#                 absErr = abs(midpoint.phi - normalizationFactor*trueWavefunction(energyLevel,midpoint.x,midpoint.y,midpoint.z))
                if absErr > maxErr:
                    maxErr = absErr
        if np.sum(errorsIfSameSign) < np.sum(errorsIfDifferentSign):
            errors = errorsIfSameSign
        else:
            errors = errorsIfDifferentSign
                   
        self.L2NormError = np.sum(errors)
        self.maxCellError = np.max(errors)
        self.maxPointwiseError = maxErr
        
    def normalizeWavefunction(self):
        """ Compute integral phi*2 dxdydz """
        A = 0.0
        maxPhi = 0.0
        
        for element in self.masterList:
            if element[1].leaf == True:
                for i,j,k in self.PxByPyByPz:
                    if abs(element[1].gridpoints[i,j,k].phi) > maxPhi:
                        maxPhi = abs(element[1].gridpoints[i,j,k].phi)
                    maxPhi = max( maxPhi, abs(element[1].gridpoints[i,j,k].phi))
                    A += element[1].gridpoints[i,j,k].phi**2*element[1].w[i,j,k]
        if A<0.0:
            print('Warning: normalization value A is less than zero...')
        if A==0.0:
            print('Warning: normalization value A is zero...')

        maxPhi=0.0        
        """ Initialize the normalization flag for each gridpoint """        
        for element in self.masterList:
            if element[1].leaf==True:
                for i,j,k in self.PxByPyByPz:
                    element[1].gridpoints[i,j,k].normalized = False
        
        """ Rescale wavefunction values, flip the flag """
        for element in self.masterList:
            if element[1].leaf==True:
                for i,j,k in self.PxByPyByPz:
                    if element[1].gridpoints[i,j,k].normalized == False:
                        element[1].gridpoints[i,j,k].phi /= np.sqrt(A)
                        element[1].gridpoints[i,j,k].normalized = True
                        maxPhi = max(maxPhi,abs(element[1].gridpoints[i,j,k].phi))
         
    def orthogonalizeWavefunction(self,n):
        """ Orthgononalizes phi against wavefunction n """
        B = 0.0
        for element in self.masterList:
            if element[1].leaf == True:
                midpoint = element[1].gridpoints[1,1,1]
                B += midpoint.phi*midpoint.finalWavefunction[n]*element[1].volume
                
        """ Initialize the orthogonalization flag for each gridpoint """        
        for element in self.masterList:
            if element[1].leaf==True:
                for i,j,k in self.PxByPyByPz:
                    element[1].gridpoints[i,j,k].orthogonalized = False
        
        """ Subtract the projection, flip the flag """
        for element in self.masterList:
            if element[1].leaf==True:
                for i,j,k in self.PxByPyByPz:
                    gridpoint = element[1].gridpoints[i,j,k]
                    if gridpoint.orthogonalized == False:
                        gridpoint.phi -= B*gridpoint.finalWavefunction[n]
                        gridpoint.orthogonalized = True
                        
    
    """
    IMPORT/EXPORT FUNCTIONS
    """                   
    def extractLeavesMidpointsOnly(self):
        '''
        Extract the leaves as a Nx4 array [ [x1,y1,z1,psi1], [x2,y2,z2,psi2], ... ]
        '''
#         leaves = np.empty((self.numberOfGridpoints,4))
        leaves = []
        counter=0
        for element in self.masterList:
            if element[1].leaf == True:
                midpoint =  element[1].gridpoints[1,1,1]
                leaves.append( [midpoint.x, midpoint.y, midpoint.z, midpoint.phi, potential(midpoint.x, midpoint.y, midpoint.z), element[1].volume ] )
                counter+=1 
                
        print('Warning: extracting midpoints even tho this is a non-uniform mesh.')
        return np.array(leaves)
    
    def extractLeavesAllGridpoints(self):
        '''
        Extract the leaves as a Nx4 array [ [x1,y1,z1,psi1], [x2,y2,z2,psi2], ... ]
        '''
#         print('Extracting the gridpoints from all leaves...')
        leaves = []
        for element in self.masterList:
            for i,j,k in self.PxByPyByPz:
                element[1].gridpoints[i,j,k].extracted = False
                
        for element in self.masterList:
            if element[1].leaf == True:
                for i,j,k in self.PxByPyByPz:
                    gridpt = element[1].gridpoints[i,j,k]
                    if gridpt.extracted == False:
                        leaves.append( [gridpt.x, gridpt.y, gridpt.z, gridpt.phi, potential(gridpt.x, gridpt.y, gridpt.z), element[1].w[i,j,k] ] )
                        gridpt.extracted = True
                    

        for element in self.masterList:
            for i,j,k in self.PxByPyByPz:
                element[1].gridpoints[i,j,k].extracted = False
                
#         print( max( abs( np.array( leaves[:][0]))))
#         print( max( abs( np.array(leaves[:][4]))))
        return np.array(leaves)
                
    def importPhiOnLeaves(self,phiNew):
        '''
        Import phi values, apply to leaves
        '''
        for element in self.masterList:
            for i,j,k in self.PxByPyByPz:
                element[1].gridpoints[i,j,k].phiImported = False
        importIndex = 0        
        for element in self.masterList:
            if element[1].leaf == True:
                for i,j,k in self.PxByPyByPz:
                    gridpt = element[1].gridpoints[i,j,k]
                    if gridpt.phiImported == False:
                        gridpt.phi = phiNew[importIndex]
                        gridpt.phiImported = True
                        importIndex += 1
                    
        for element in self.masterList:
            for i,j,k in self.PxByPyByPz:
                element[1].gridpoints[i,j,k].phiImported = None
        if importIndex != len(phiNew):
            print('Warning: import index not equal to len(phiNew)')
            print(importIndex)
            print(len(phiNew))
                
    def copyPhiToFinalOrbital(self, n):
        for element in self.masterList:
            for i,j,k in self.PxByPyByPz:
                gridpt = element[1].gridpoints[i,j,k]
                if len(gridpt.finalWavefunction) == n:
                    gridpt.finalWavefunction.append(gridpt.phi)
                     
    def exportMeshVTK(self,filename):
        def mkVtkIdList(it): # helper function
            vil = vtk.vtkIdList()
            for i in it:
                vil.InsertNextId(int(i))
            return vil
        
        mesh    = vtk.vtkPolyData()
        points  = vtk.vtkPoints()
        polys   = vtk.vtkCellArray()
        scalars = vtk.vtkFloatArray()
        
        coords = []
        faces = []
        pointcounter=0
        for element in self.masterList:
            cell = element[1]
            if cell.leaf == True:
                coords.append( (cell.xmin, cell.ymin, cell.zmin))
                coords.append( (cell.xmax, cell.ymin, cell.zmin))
                coords.append( (cell.xmax, cell.ymax, cell.zmin))
                coords.append( (cell.xmin, cell.ymax, cell.zmin))
                coords.append( (cell.xmin, cell.ymin, cell.zmax))
                coords.append( (cell.xmax, cell.ymin, cell.zmax))
                coords.append( (cell.xmax, cell.ymax, cell.zmax))
                coords.append( (cell.xmin, cell.ymax, cell.zmax))
                
                for i in range(8):
                    scalars.InsertTuple1(pointcounter+i,cell.level)
                
                faces.append((pointcounter+0,pointcounter+1,pointcounter+2,pointcounter+3))
                faces.append((pointcounter+4,pointcounter+5,pointcounter+6,pointcounter+7))
                faces.append((pointcounter+0,pointcounter+1,pointcounter+5,pointcounter+4))
                faces.append((pointcounter+1,pointcounter+2,pointcounter+6,pointcounter+5))
                faces.append((pointcounter+2,pointcounter+3,pointcounter+7,pointcounter+6))
                faces.append((pointcounter+3,pointcounter+0,pointcounter+4,pointcounter+7))

                pointcounter+=8
        
#         atomcounter=0
#         for atom in self.atoms:
#             coords.append( (atom.x, atom.y, atom.z) )
# #             scalars.InsertTuple1(pointcounter+atomcounter,-1)
#             atomcounter+=1
                
        for i in range(len(coords)):
            points.InsertPoint(i, coords[i])
        for i in range(len(faces)):
            polys.InsertNextCell( mkVtkIdList(faces[i]) )
            
        mesh.SetPoints(points)
        mesh.SetPolys(polys)
        mesh.GetPointData().SetScalars(scalars)

        writer = vtk.vtkPolyDataWriter()
        writer.SetFileName(filename)
      
        writer.SetInputData(mesh)

        writer.Write()
        print('Done writing ', filename)
    
                                    
def TestTreeForProfiling():
    xmin = ymin = zmin = -12
    xmax = ymax = zmax = -xmin
    tree = Tree(xmin,xmax,ymin,ymax,zmin,zmax)
    tree.buildTree( minLevels=4, maxLevels=4, divideTolerance=0.07,printTreeProperties=True)
    
               
if __name__ == "__main__":
    TestTreeForProfiling()
    

    
    
       
    