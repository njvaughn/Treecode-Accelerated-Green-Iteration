'''
The main Tree data structure.  The root of the tree is a Cell object that is comprised of the 
entire domain.  The tree gets built by dividing the root cell, recursively, based on the set 
divide condition.  The current implementation uses the variation of phi within a cell to 
dictate whether or not it divides.  

Cells can perform recursive functions on the tree.  The tree can also extract all gridpoints or
all midpoints as arrays which can be fed in to the GPU kernels, or other tree-external functions.
-- 03/20/2018 NV

@author: nathanvaughn
'''

import numpy as np
from scipy.interpolate import interp1d
import pylibxc
import itertools
import os
import csv
import vtk
try:
    from pyevtk.hl import pointsToVTK
except ImportError:
    pass

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
    def __init__(self, xmin,xmax,px,ymin,ymax,py,zmin,zmax,pz,nElectrons=2,nOrbitals=1,coordinateFile='',xcFunctional="LDA_XC_LP_A",polarization="unpolarized", printTreeProperties = True):
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
        self.nElectrons = nElectrons
        self.nOrbitals = nOrbitals
        
        
        self.xcFunc = pylibxc.LibXCFunctional(xcFunctional, polarization)
        self.orbitalEnergies = -np.ones(nOrbitals)
        
        # generate gridpoint objects.  
        xvec = ChebyshevPoints(self.xmin,self.xmax,self.px)
        yvec = ChebyshevPoints(self.ymin,self.ymax,self.py)
        zvec = ChebyshevPoints(self.zmin,self.zmax,self.pz)
        gridpoints = np.empty((px,py,pz),dtype=object)

        for i, j, k in self.PxByPyByPz:
            gridpoints[i,j,k] = GridPoint(xvec[i],yvec[j],zvec[k],self.nOrbitals)
        
        # generate root cell from the gridpoint objects  
        self.root = Cell( self.xmin, self.xmax, self.px, 
                          self.ymin, self.ymax, self.py, 
                          self.zmin, self.zmax, self.pz, 
                          gridpoints, self )
        self.root.level = 0
        self.root.uniqueID = ''
        self.masterList = [[self.root.uniqueID, self.root]]
        
        self.initialDivideBasedOnNuclei(coordinateFile)
        if  printTreeProperties == True:
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
        
        print('Searching for atom data in: ', coordinateFile)
        atomData = np.genfromtxt(coordinateFile,delimiter=',',dtype=float)
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
        
        self.computeNuclearNuclearEnergy()
        for atom in self.atoms:
            recursiveDivideByAtom(self,atom,self.root)
        
#         self.exportMeshVTK('/Users/nathanvaughn/Desktop/aspectRatioBefore2.vtk')
        for _,cell in self.masterList:
            if cell.leaf==True:
                cell.divideIfAspectRatioExceeds(2.0)
      

    def initializeFromAtomicData(self):
        # Generalized for any atoms.  Not complete yet.  
        timer = Timer()
        timer.start()
        
        interpolators = np.empty(self.nOrbitals,dtype=object)
        count=0
        for atom in self.atoms:
            path = '/Users/nathanvaughn/AtomicData/allElectron/z'+str(int(atom.atomicNumber))+'/singleAtomData/'
#             print('Searching for single atom data in: ',path)
            for orbital in os.listdir(path): 
                if orbital[:3]=='psi':
                    data = np.genfromtxt(path+orbital)
                    interpolators[count] = interp1d(data[:,0],data[:,1])
                    count+=1
            
            for _,cell in self.masterList:
                for i,j,k in self.PxByPyByPz:
                    # compute radius, as well as polar and asimuthal angles when the time comes...
                    gp = cell.gridpoints[i,j,k]
                    r = np.sqrt(gp.x*gp.x + gp.y*gp.y + gp.z*gp.z )
                    
                    # figure out teh mapping between orbitals and interpolators.  For instance, the 
                    # 2p interpolator needs to be used for three orbitals, each with a different spherical harmonic.  
                    # maybe keep a counter, and loop over the allowed values of ell, only incrementing the counter
                    # when all values of ell are exhausted.  But need the mapping from n to s.  The data file name has
                    # the numbers I need.  'psi32.inp' for example.  
                    
                    counter=0
                    for orbitalNumber in range(self.nOrbitals):
                        for orbital in os.listdir(path):
                            if orbital[:3]=='psi':
                                ell = orbital[4]
                        
                        for ii in range(ell):
                            pass

                        counter+=1
                            
                        
                    
        timer.stop()
        print('Initialization from single atom data took %f.3 seconds.' %timer.elapsedTime)
        
    def initializeForBerylliumAtom(self):
        print('Initializing orbitals for beryllium atom exclusively. ')

        # Generalized for any atoms.  Not complete yet.  
        timer = Timer()
        timer.start()
        
        interpolators = np.empty(self.nOrbitals,dtype=object)
        count=0
        BerylliumAtom = self.atoms[0]
#         path = '/Users/nathanvaughn/AtomicData/allElectron/z'+str(BerylliumAtom.atomicNumber)+'/singleAtomData/'
        path = '/home/njvaughn/AtomicData/allElectron/z'+str(BerylliumAtom.atomicNumber)+'/singleAtomData/'
        for orbital in os.listdir(path): 
            if orbital[:5]=='psi10':
                data = np.genfromtxt(path+orbital)
                interpolators[0] = interp1d(data[:,0],data[:,1])
            if orbital[:5]=='psi20':
                data = np.genfromtxt(path+orbital)
                interpolators[1] = interp1d(data[:,0],data[:,1])
            
        for _,cell in self.masterList:
            for i,j,k in self.PxByPyByPz:
                gp = cell.gridpoints[i,j,k]
                r = np.sqrt( (gp.x-BerylliumAtom.x)**2 + (gp.y-BerylliumAtom.y)**2 + (gp.z-BerylliumAtom.z)**2 )
                
                gp.setPhi(interpolators[0](r),0)
                gp.setPhi(interpolators[1](r),1)
                   
        timer.stop()
        print('Initialization from single Beryllium atom data took %f.3 seconds.' %timer.elapsedTime)
        
    def initializeForHydrogenMolecule(self):
        print('Initializing orbitals for hydrogen molecule exclusively. ')
        # Generalized for any atoms.  Not complete yet.  
        timer = Timer()
        timer.start()
        
        for atom in self.atoms:
            interpolators = np.empty((self.nOrbitals,),dtype=object) 
            count=0
#             path = '/Users/nathanvaughn/AtomicData/allElectron/z'+str(atom.atomicNumber)+'/singleAtomData/'
            path = '/home/njvaughn/AtomicData/allElectron/z'+str(atom.atomicNumber)+'/singleAtomData/'
            for orbital in os.listdir(path): 
                if orbital[:5]=='psi10':
                    data = np.genfromtxt(path+orbital)
                    interpolators[count] = interp1d(data[:,0],data[:,1])  # issue: data for multiple orbitals, but I only have 1.  
                    count+=1
                
            for _,cell in self.masterList:
                for i,j,k in self.PxByPyByPz:
                    gp = cell.gridpoints[i,j,k]
                    r = np.sqrt( (gp.x-atom.x)**2 + (gp.y-atom.y)**2 + (gp.z-atom.z)**2 )
#                     print(interpolators[0](r))
                    gp.setPhi(gp.phi[0] + interpolators[0](r),0)
                   
        timer.stop()
        print('Initialization from single atom data took %f.3 seconds.' %timer.elapsedTime)
        
        
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
                    Cell.divide(xdiv, ydiv, zdiv, printNumberOfCells)
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
        
        """ Count the number of unique leaf cells and gridpoints and set initial external potential """
        self.numberOfGridpoints = 0
        self.numberOfCells = 0
        for _,cell in self.masterList:
            if cell.leaf==True:
                self.numberOfCells += 1
                for i,j,k in self.PxByPyByPz:
                    if not hasattr(cell.gridpoints[i,j,k], "counted"):
                        self.numberOfGridpoints += 1
                        cell.gridpoints[i,j,k].counted = True
                        cell.gridpoints[i,j,k].setExternalPotential(self.atoms)
        
                        
        for _,cell in self.masterList:
            for i,j,k in self.PxByPyByPz:
                if hasattr(cell.gridpoints[i,j,k], "counted"):
                    cell.gridpoints[i,j,k].counted = None
         
        
#         print('Initializing phi to hydrogen atom wavefunctions.')            
#         print('Initializing phi to superposition of single atom ground states.')            
#         for _,cell in self.masterList:
#             for i,j,k in cell.PxByPyByPz:
#                 gp = cell.gridpoints[i,j,k]
#                 r1 = np.sqrt((gp.x-0.7)*(gp.x-0.7) + gp.y*gp.y + gp.z*gp.z)
#                 r2 = np.sqrt((gp.x+0.7)*(gp.x+0.7) + gp.y*gp.y + gp.z*gp.z)
# 
# #                 r = np.sqrt(gp.x*gp.x + gp.y*gp.y + gp.z*gp.z)
#                 for m in range(self.nOrbitals):
#                     gp.phi[m] = np.exp(-r1) + np.exp(-r2)
# #                     gp.phi[m] = np.exp(-4*r)*r**m  
# 
# #                 r = np.sqrt(gp.x*gp.x + gp.y*gp.y + gp.z*gp.z)
# #                 gp.phi = np.exp(-r)

#         self.initializeFromAtomicData()
        self.initializeForHydrogenMolecule()
#         self.initializeForBerylliumAtom()
        self.orthonormalizeOrbitals()
        
        
                    
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
        # outdated to when I had analytic wavaefunction for Hydrogen atom
        for element in self.masterList:
            for i,j,k in self.PxByPyByPz:
                element[1].gridpoints[i,j,k].setAnalyticPhi(n)
        self.normalizeOrbital()   
    
    def populatePhi(self):
        for element in self.masterList:
            for i,j,k in self.PxByPyByPz:
                gp=element[1].gridpoints[i,j,k]
                for m in range(self.nOrbitals):
                    gp.setPhi(np.exp( - np.sqrt( gp.x**2 + gp.y**2 + gp.z**2 )), m)
        print('Populating phi from tree.populatePhi()')
        self.orthonormalizeOrbitals()  
        
    def refineOnTheFly(self, divideTolerance):
        counter = 0
        for _,cell in self.masterList:
            if cell.leaf==True:
                rhoVariation = cell.getRhoVariation()
                if rhoVariation > divideTolerance:
                    cell.divide(cell.xmid, cell.ymid, cell.zmid,interpolate=True)
                    counter+=1
        print('Refined %i cells.' %counter)
            

    
    """
    UPDATE DENSITY AND EFFECTIVE POTENTIAL AT GRIDPOINTS
    """
    def updateVxcAndVeffAtQuadpoints(self):
#         print('Warning: v_xc is zeroed out')
        
        def CellupdateVxcAndVeff(cell,xcFunc):
            '''
            After density is updated the convolution gets called to update V_coulom.
            Now I need to update v_xc, then get the new value of v_eff. 
            '''
            
            rho = np.empty((cell.px,cell.py,cell.pz))
            
            for i,j,k in cell.PxByPyByPz:
                rho[i,j,k] = cell.gridpoints[i,j,k].rho
                
            xcOutput = xcFunc.compute(rho)
            EXC = np.reshape(xcOutput['zk'],np.shape(rho))
            VRHO = np.reshape(xcOutput['vrho'],np.shape(rho))
            
            for i,j,k in self.PxByPyByPz:
                cell.gridpoints[i,j,k].epsilon_xc = EXC[i,j,k]
                cell.gridpoints[i,j,k].v_xc = VRHO[i,j,k]
#                 cell.gridpoints[i,j,k].v_xc = 0
                cell.gridpoints[i,j,k].updateVeff()
            
        for _,cell in self.masterList:
            if cell.leaf == True:
                CellupdateVxcAndVeff(cell,self.xcFunc)

    def updateDensityAtQuadpoints(self):
        def CellUpdateDensity(cell):
            for i,j,k in self.PxByPyByPz:
                for m in range(self.nOrbitals):
                    cell.gridpoints[i,j,k].rho = cell.gridpoints[i,j,k].phi[m]**2
        
        for _,cell in self.masterList:
            if cell.leaf==True:
                CellUpdateDensity(cell)
     
    def normalizeDensity(self):            
        def integrateDensity(cell):
            rho = np.empty((cell.px,cell.py,cell.pz))
                        
            for i,j,k in cell.PxByPyByPz:
                gp = cell.gridpoints[i,j,k]
                rho[i,j,k] = gp.rho
            
            return np.sum( cell.w * rho )
        
        B = 0.0
        for _,cell in self.masterList:
            if cell.leaf == True:
                B += integrateDensity(cell)
        if B==0.0:
            print('Warning, integrated density to 0')
#         print('Raw computed density ', B)
        for _,cell in self.masterList:
            if cell.leaf == True:
                for i,j,k in cell.PxByPyByPz:
                    cell.gridpoints[i,j,k].rho/=(B/self.nElectrons)
#         B = 0.0
#         for id,cell in self.masterList:
#             if cell.leaf == True:
#                 B += integrateDensity(cell)
#         print('Integrated density after normalization ', B)
    
                
            
    
    """
    ENERGY COMPUTATION FUNCTIONS
    """       
    def computeTotalPotential(self): 
        
        def integrateCellDensityAgainst__(cell,integrand):
            rho = np.empty((cell.px,cell.py,cell.pz))
            pot = np.empty((cell.px,cell.py,cell.pz))
            
            for i,j,k in cell.PxByPyByPz:
                gp = cell.gridpoints[i,j,k]
                rho[i,j,k] = gp.rho
                pot[i,j,k] = getattr(gp,integrand)
            
            return np.sum( cell.w * rho * pot) 
        
        V_xc = 0.0
        V_coulomb = 0.0
        E_xc = 0.0
        
        for _,cell in self.masterList:
            if cell.leaf == True:
                V_xc += integrateCellDensityAgainst__(cell,'v_xc')
                V_coulomb += integrateCellDensityAgainst__(cell,'v_coulomb')
                E_xc += integrateCellDensityAgainst__(cell,'epsilon_xc')
        self.totalVxc = V_xc
        self.totalVcoulomb = V_coulomb
        self.totalExc = E_xc
#         print('Total V_xc : ')
        
#         self.totalPotential = -1/2*V_coulomb + E_xc - V_xc 
        self.totalPotential = -1/2*V_coulomb + E_xc - V_xc + self.nuclearNuclear
                
        
       
    def computeTotalKinetic(self):
        # sum over the kinetic energies of all orbitals
        self.totalKinetic = 0.0
        for i in range(self.nOrbitals):
            self.totalKinetic += 2*self.orbitalEnergies[i]  # this factor of 2 is in question
        
    
    def updateTotalEnergy(self):
        self.computeTotalKinetic()
        self.computeTotalPotential()
        self.E = self.totalKinetic + self.totalPotential
    
    def computeOrbitalPotentials(self): 
        
        self.orbitalPotential = np.zeros(self.nOrbitals)  
        for _,cell in self.masterList:
            if cell.leaf == True:
                cell.computeOrbitalPotentials()
                self.orbitalPotential += cell.orbitalPE
                       
    def computeOrbitalKinetics(self):

        self.orbitalKinetic = np.zeros(self.nOrbitals)
        for _,cell in self.masterList:
            if cell.leaf == True:
                cell.computeOrbitalKinetics()
                self.orbitalKinetic += cell.orbitalKE
            
        
    def updateOrbitalEnergies(self):
        self.computeOrbitalKinetics()
        self.computeOrbitalPotentials()
        print('Orbital Kinetic Energy:   ', self.orbitalKinetic)
        print('Orbital Potential Energy: ', self.orbitalPotential)
        self.orbitalEnergies = self.orbitalKinetic + self.orbitalPotential
        for m in range(self.nOrbitals):
            if self.orbitalEnergies[m] > 0:
                print('Warning, orbital energy is positive.  Resetting to -0.5')
                self.orbitalEnergies[m] = -0.5
            
    def computeNuclearNuclearEnergy(self):
        self.nuclearNuclear = 0.0
        for atom1 in self.atoms:
            for atom2 in self.atoms:
                if atom1!=atom2:
                    r = sqrt( (atom1.x-atom2.x)**2 + (atom1.y-atom2.y)**2 + (atom1.z-atom2.z)**2 )
                    self.nuclearNuclear += atom1.atomicNumber*atom2.atomicNumber/r
        self.nuclearNuclear /= 2 # because of double counting
        print('Nuclear energy: ', self.nuclearNuclear)
            
                    
    """
    NORMALIZATION, ORTHOGONALIZATION, AND WAVEFUNCTION ERRORS
    """      
    def computeWaveErrors(self,energyLevel,normalizationFactor):
        # outdated from when I had analytic wavefunctions for hydrogen atom
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
        
    def normalizeOrbital(self):
        """ Enforce integral phi*2 dxdydz == 1 """
        A = 0.0
#         maxPhi = 0.0
        
        for element in self.masterList:
            if element[1].leaf == True:
                for i,j,k in self.PxByPyByPz:
#                     if abs(element[1].gridpoints[i,j,k].phi) > maxPhi:
#                         maxPhi = abs(element[1].gridpoints[i,j,k].phi)
#                     maxPhi = max( maxPhi, abs(element[1].gridpoints[i,j,k].phi))
                    A += element[1].gridpoints[i,j,k].phi**2*element[1].w[i,j,k]
        if A<0.0:
            print('Warning: normalization value A is less than zero...')
        if A==0.0:
            print('Warning: normalization value A is zero...')

#         maxPhi=0.0        
#         """ Initialize the normalization flag for each gridpoint """        
#         for element in self.masterList:
#             if element[1].leaf==True:
#                 for i,j,k in self.PxByPyByPz:
#                     element[1].gridpoints[i,j,k].normalized = False
        
        """ Rescale wavefunction values, flip the flag """
        for element in self.masterList:
            if element[1].leaf==True:
                for i,j,k in self.PxByPyByPz:
                        element[1].gridpoints[i,j,k].phi /= np.sqrt(A)
#                     if element[1].gridpoints[i,j,k].normalized == False:
#                         element[1].gridpoints[i,j,k].phi /= np.sqrt(A)
#                         element[1].gridpoints[i,j,k].normalized = True
#                         maxPhi = max(maxPhi,abs(element[1].gridpoints[i,j,k].phi))
           
         
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
                        
    def orthonormalizeOrbitals(self):
        
        def orthogonalizeOrbitals(tree,m,n):
            print('Orthogonalizing orbital %i against %i' %(m,n))
            """ Compute the overlap, integral phi_r * phi_s """
            B = 0.0
            for _,cell in tree.masterList:
                if cell.leaf == True:
                    for i,j,k in self.PxByPyByPz:
                        phi_m = cell.gridpoints[i,j,k].phi[m]
                        phi_n = cell.gridpoints[i,j,k].phi[n]
                        B += phi_m*phi_n*cell.w[i,j,k]
                    
            """ Subtract the projection """
            for _,cell in tree.masterList:
                if cell.leaf==True:
                    for i,j,k in self.PxByPyByPz:
                        gridpoint = cell.gridpoints[i,j,k]
                        gridpoint.phi[m] -= B*gridpoint.phi[n]
        
        def normalizeOrbital(tree,m):
        
            """ Enforce integral phi*2 dxdydz == 1 """
            A = 0.0        
            for _,cell in tree.masterList:
                if cell.leaf == True:
                    for i,j,k in self.PxByPyByPz:
                        A += cell.gridpoints[i,j,k].phi[m]**2*cell.w[i,j,k]
    
            """ Rescale wavefunction values, flip the flag """
            for _,cell in tree.masterList:
                if cell.leaf==True:
                    for i,j,k in self.PxByPyByPz:
                            cell.gridpoints[i,j,k].phi /= np.sqrt(A)
        
        for m in range(self.nOrbitals):
            for n in range(m):
                orthogonalizeOrbitals(self,m,n)
            normalizeOrbital(self,m)
            
            
    
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
        for _,cell in self.masterList:
            for i,j,k in self.PxByPyByPz:
                cell.gridpoints[i,j,k].extracted = False
                
        for _,cell in self.masterList:
            if cell.leaf == True:
                for i,j,k in self.PxByPyByPz:
                    gridpt = cell.gridpoints[i,j,k]
                    if gridpt.extracted == False:
                        leaves.append( [gridpt.x, gridpt.y, gridpt.z, gridpt.phi, gridpt.v_eff, cell.w[i,j,k] ] )
                        gridpt.extracted = True
                    

        for _,cell in self.masterList:
            for i,j,k in self.PxByPyByPz:
                cell.gridpoints[i,j,k].extracted = False
                
        return np.array(leaves)
    
    def extractPhi(self, orbitalNumber):
        '''
        Extract the leaves as a Nx4 array [ [x1,y1,z1,psi1], [x2,y2,z2,psi2], ... ]
        '''
#         print('Extracting the gridpoints from all leaves...')
        leaves = []
#         for _,cell in self.masterList:
#             for i,j,k in self.PxByPyByPz:
#                 cell.gridpoints[i,j,k].extracted = False
                
        for _,cell in self.masterList:
            if cell.leaf == True:
                for i,j,k in self.PxByPyByPz:
                    gridpt = cell.gridpoints[i,j,k]
#                     if gridpt.extracted == False:
                    leaves.append( [gridpt.x, gridpt.y, gridpt.z, gridpt.phi[orbitalNumber], gridpt.v_eff, cell.w[i,j,k] ] )
#                         gridpt.extracted = True
                    

#         for _,cell in self.masterList:
#             for i,j,k in self.PxByPyByPz:
#                 cell.gridpoints[i,j,k].extracted = False
                
        return np.array(leaves)
    
    def extractLeavesDensity(self):
        '''
        Extract the leaves as a Nx5 array [ [x1,y1,z1,rho1,w1], [x2,y2,z2,rho2,w2], ... ]
        '''
#         print('Extracting the gridpoints from all leaves...')
        leaves = []
                
        for element in self.masterList:
            if element[1].leaf == True:
                for i,j,k in self.PxByPyByPz:
                    gridpt = element[1].gridpoints[i,j,k]
                    leaves.append( [gridpt.x, gridpt.y, gridpt.z, gridpt.rho, element[1].w[i,j,k] ] )
                            
        return np.array(leaves)
                
    def importPhiOnLeaves(self,phiNew, orbitalNumber):
        '''
        Import phi values, apply to leaves
        '''
#         for _,cell in self.masterList:
#             for i,j,k in self.PxByPyByPz:
#                 cell.gridpoints[i,j,k].phiImported = False
        importIndex = 0        
        for _,cell in self.masterList:
            if cell.leaf == True:
                for i,j,k in self.PxByPyByPz:
                    gridpt = cell.gridpoints[i,j,k]
#                     if gridpt.phiImported == False:
                    gridpt.phi[orbitalNumber] = phiNew[importIndex]
#                         gridpt.phiImported = True
                    importIndex += 1
                    
#         for _,cell in self.masterList:
#             for i,j,k in self.PxByPyByPz:
#                 cell.gridpoints[i,j,k].phiImported = None
        if importIndex != len(phiNew):
            print('Warning: import index not equal to len(phiNew)')
            print(importIndex)
            print(len(phiNew))
            
    def importVcoulombOnLeaves(self,V_coulombNew):
        '''
        Import V_coulomng values, apply to leaves
        '''

        importIndex = 0        
        for element in self.masterList:
            if element[1].leaf == True:
                for i,j,k in self.PxByPyByPz:
                    gridpt = element[1].gridpoints[i,j,k]
                    gridpt.v_coulomb = V_coulombNew[importIndex]
                    importIndex += 1

        if importIndex != len(V_coulombNew):
            print('Warning: import index not equal to len(V_coulombNew)')
            print(importIndex)
            print(len(V_coulombNew))
            
    
                
    def copyPhiToFinalOrbital(self, n):
        # outdated
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
                
#                 x = np.empty(cell.px)
#                 y = np.empty(cell.py)
#                 z = np.empty(cell.pz)
#                 for i in range(cell.px):
#                     x[i] = cell.gridpoints[i,0,0].x
#                 for j in range(cell.py):
#                     y[j] = cell.gridpoints[0,j,0].y
#                 for k in range(cell.pz):
#                     z[k] = cell.gridpoints[0,0,k].z
#                     
#                 
#                 # Generate interpolators for each orbital
#                 interpolators = np.empty(self.nOrbitals,dtype=object)
#                 phi = np.zeros((cell.px,cell.py,cell.pz,self.nOrbitals))
#                 for i,j,k in self.PxByPyByPz:
#                     for m in range(self.nOrbitals):
#                         phi[i,j,k,m] = cell.gridpoints[i,j,k].phi[m]
#                 
#                 for m in range(self.nOrbitals):
#                     interpolators[m] = cell.interpolator(x, y, z, phi[:,:,:,m])
                
                for i in range(8):
#                     scalars.InsertTuple1(pointcounter+i,cell.level)
#                     scalars.InsertTuple1(pointcounter+i,interpolators[0](cell.xmid, cell.ymid, cell.zmid))
                    scalars.InsertTuple1(pointcounter+i,cell.gridpoints[1,1,1].phi)
                
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
        
    def exportGridpoints(self,filename):
        x = []
        y = []
        z = []
        v = []
        phi10 = []
        phi20 = []
        for _,cell in self.masterList:
            if cell.leaf==True:
                for i,j,k in cell.PxByPyByPz:
                    gp = cell.gridpoints[i,j,k]
                    x.append(gp.x)
                    y.append(gp.y)
                    z.append(gp.z)
                    v.append(gp.v_eff)
                    phi10.append(gp.phi[0])
                    phi20.append(gp.phi[1])
        
        pointsToVTK(filename, np.array(x), np.array(y), np.array(z), data = 
                    {"V" : np.array(v), "Phi10" : np.array(phi10), "Phi20" : np.array(phi20)})
                
        
    def exportGreenIterationOrbital(self,filename,iterationNumber):
        def mkVtkIdList(it): # helper function
            vil = vtk.vtkIdList()
            for i in it:
                vil.InsertNextId(int(i))
            return vil
        
        mesh    = vtk.vtkPolyData()
        points  = vtk.vtkPoints()
#         polys   = vtk.vtkCellArray()
        scalars = vtk.vtkFloatArray()
        
        coords = []
        faces = []
        pointcounter=0
        for element in self.masterList:
            cell = element[1]
            if cell.leaf == True:
                phi = 0.0
                # compute representative phi
                for i,j,k in cell.PxByPyByPz:
                    phi += cell.gridpoints[i,j,k].phi*cell.w[i,j,k]
                
                coords.append( (cell.xmid,cell.ymid,cell.zmid,iterationNumber) )
                scalars.InsertTuple1(pointcounter,cell.level)
                
#                 for i in range(8):
#                     scalars.InsertTuple1(pointcounter+i,phi)
#                 
#                 faces.append((pointcounter+0,pointcounter+1,pointcounter+2,pointcounter+3))
#                 faces.append((pointcounter+4,pointcounter+5,pointcounter+6,pointcounter+7))
#                 faces.append((pointcounter+0,pointcounter+1,pointcounter+5,pointcounter+4))
#                 faces.append((pointcounter+1,pointcounter+2,pointcounter+6,pointcounter+5))
#                 faces.append((pointcounter+2,pointcounter+3,pointcounter+7,pointcounter+6))
#                 faces.append((pointcounter+3,pointcounter+0,pointcounter+4,pointcounter+7))

                pointcounter+=1
        
#         atomcounter=0
#         for atom in self.atoms:
#             coords.append( (atom.x, atom.y, atom.z) )
# #             scalars.InsertTuple1(pointcounter+atomcounter,-1)
#             atomcounter+=1
                
        for i in range(len(coords)):
            points.InsertPoint(i, coords[i])
#         for i in range(len(faces)):
#             polys.InsertNextCell( mkVtkIdList(faces[i]) )
            
        mesh.SetPoints(points)
#         mesh.SetPolys(polys)
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
    

    
    
       
    