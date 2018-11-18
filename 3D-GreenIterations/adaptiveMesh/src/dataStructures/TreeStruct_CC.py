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
from scipy.special import sph_harm
from scipy.optimize import broyden1, anderson, brentq
import pylibxc
import itertools
import os
import csv
try:
    import vtk
except ModuleNotFoundError:
    pass
import time
import copy
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

psiList = ['psi10', 'psi20', 'psi21', 'psi30', 'psi32']

class Tree(object):
    '''
    Tree object. Constructed of cells, which are composed of gridpoint objects.  
    Trees contain their root, as well as their masterList.
    '''
        
    """
    INTIALIZATION FUNCTIONS
    """
    
    
    def __init__(self, xmin,xmax,px,ymin,ymax,py,zmin,zmax,pz,nElectrons,nOrbitals,maxDepthAtAtoms,gaugeShift=0.0,
                 coordinateFile='',inputFile='',exchangeFunctional="LDA_X",correlationFunctional="LDA_C_PZ",
                 polarization="unpolarized", 
                 printTreeProperties = True):
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
        self.gaugeShift = gaugeShift
        self.maxDepthAtAtoms = maxDepthAtAtoms
        
        self.mixingParameter=0.75  # (1-mixingParam)*rhoNew
#         self.mixingParameter=-1 # accelerate with -1
#         self.occupations = np.ones(nOrbitals)
#         self.computeOccupations()
        
        
        self.exchangeFunctional = pylibxc.LibXCFunctional(exchangeFunctional, polarization)
        self.correlationFunctional = pylibxc.LibXCFunctional(correlationFunctional, polarization)
        
        self.orbitalEnergies = -np.ones(nOrbitals)
        
        print('Reading atomic coordinates from: ', coordinateFile)
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
        
        # generate gridpoint objects.  
        xvec = ChebyshevPoints(self.xmin,self.xmax,self.px)
        yvec = ChebyshevPoints(self.ymin,self.ymax,self.py)
        zvec = ChebyshevPoints(self.zmin,self.zmax,self.pz)
        gridpoints = np.empty((px,py,pz),dtype=object)

        for i, j, k in self.PxByPyByPz:
            gridpoints[i,j,k] = GridPoint(xvec[i],yvec[j],zvec[k],self.nOrbitals, self.gaugeShift, self.atoms,initPotential=False)
        
        # generate root cell from the gridpoint objects  
        self.root = Cell( self.xmin, self.xmax, self.px, 
                          self.ymin, self.ymax, self.py, 
                          self.zmin, self.zmax, self.pz, 
                          gridpoints, densityPoints=None, tree=self )
        self.root.level = 0
        self.root.uniqueID = ''
        self.masterList = [[self.root.uniqueID, self.root]]
        
# #         self.gaugeShift = np.genfromtxt(inputFile,dtype=[(str,str,int,int,float,float,float,float,float)])[8]
#         self.gaugeShift = np.genfromtxt(inputFile,dtype=[(str,str,int,int,float,float,float,float,float)])[8]
        print('Gauge shift ', self.gaugeShift)
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
        
    
    
    def fermiObjectiveFunction(self,fermiEnergy):
            exponentialArg = (self.orbitalEnergies-fermiEnergy)/self.sigma
            temp = 1/(1+np.exp( exponentialArg ) )
            return self.nElectrons - 2 * np.sum(temp)
        
        
    def computeOccupations(self):
        
        self.T = 100
        KB = 8.6173303e-5/27.211386
        self.sigma = self.T*KB
        
        
        
        eF = brentq(self.fermiObjectiveFunction, self.orbitalEnergies[0], 1, xtol=1e-14)
#         eF = brentq(self.fermiObjectiveFunction, self.orbitalEnergies[0], 1)
        print('Fermi energy: ', eF)
        exponentialArg = (self.orbitalEnergies-eF)/self.sigma
        self.occupations = 2*1/(1+np.exp( exponentialArg ) )  # these are # of electrons, not fractional occupancy.  Hence the 2*
        print('Occupations: ', self.occupations)
        

            
    def initialDivideBasedOnNuclei(self, coordinateFile,maxLevels=15):
            
        def refineToMaxDepth(self,Atom,Cell):
            if hasattr(Cell, "children"):
                (ii,jj,kk) = np.shape(Cell.children)
                for i in range(ii):
                    for j in range(jj):
                        for k in range(kk):
                            if ( (Atom.x <= Cell.children[i,j,k].xmax) and (Atom.x >= Cell.children[i,j,k].xmin) ):
                                if ( (Atom.y <= Cell.children[i,j,k].ymax) and (Atom.y >= Cell.children[i,j,k].ymin) ):
                                    if ( (Atom.z <= Cell.children[i,j,k].zmax) and (Atom.z >= Cell.children[i,j,k].zmin) ): 
#                                         print('Calling refine on cell ',Cell.children[i,j,k].uniqueID)                                           
                                        refineToMaxDepth(self, Atom, Cell.children[i,j,k])
            
            else:  # cell is a leaf
                if Cell.level < self.maxDepthAtAtoms:
                    xdiv = Cell.xmid
                    ydiv = Cell.ymid
                    zdiv = Cell.zmid
#                     if ( (Atom.x == Cell.xmax) or (Atom.x == Cell.xmin) ):
#                         xdiv = None
#                     if ( (Atom.y == Cell.ymax) or (Atom.y == Cell.ymin) ):
#                         ydiv = None
#                     if ( (Atom.z == Cell.zmax) or (Atom.z == Cell.zmin) ):
#                         zdiv = None
                    Cell.divide(xdiv, ydiv, zdiv)
#                     print('Dividing cell ', Cell.uniqueID, ' at depth ', Cell.level)
                    refineToMaxDepth(self,Atom,Cell)
                
                else: 
                    # This nucleus is at the corner of this cell, but this cell is already at max depth.  Try things here
                    # such as setting the weights to zero for this cell.  
                    print('Setting weights to zero for cell ', Cell.uniqueID)
                    Cell.w = np.zeros( (Cell.px,Cell.py,Cell.pz) )
                    
                    
            
        
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
        
#         print('Reading atomic coordinates from: ', coordinateFile)
#         atomData = np.genfromtxt(coordinateFile,delimiter=',',dtype=float)
# #         print(np.shape(atomData))
# #         print(len(atomData))
#         if np.shape(atomData)==(4,):
#             self.atoms = np.empty((1,),dtype=object)
#             atom = Atom(atomData[0],atomData[1],atomData[2],atomData[3])
#             self.atoms[0] = atom
#         else:
#             self.atoms = np.empty((len(atomData),),dtype=object)
#             for i in range(len(atomData)):
#                 atom = Atom(atomData[i,0],atomData[i,1],atomData[i,2],atomData[i,3])
#                 self.atoms[i] = atom
#                 self.atoms[i] = atom
        
        self.computeNuclearNuclearEnergy()
        self.nAtoms = 0
        for atom in self.atoms:
            recursiveDivideByAtom(self,atom,self.root)
            self.nAtoms += 1
        
        
        
#         self.exportMeshVTK('/Users/nathanvaughn/Desktop/aspectRatioBefore2.vtk')
        for _,cell in self.masterList:
            if cell.leaf==True:
                cell.divideIfAspectRatioExceeds(1.5) #283904 for aspect ratio 1.5, but 289280 for aspect ratio 10.0.  BUT, for 9.5, 8, 4, and so on, there are less quad points than 2.0.  So maybe not a bug 
        
        # Reset all cells to level 1.  These divides shouldnt count towards its depth.  
        for _,cell in self.masterList:
            if cell.leaf==True:
                cell.level = 0
        
#         print('Dividing adjacent to nuclei')  
#         for atom in self.atoms:
#             refineToMaxDepth(self,atom,self.root)
                
      
    def initializeOrbitalsRandomly(self):
        print('Initializing orbitals randomly...')
        for _,cell in self.masterList:
            if cell.leaf==True:
                for i,j,k in self.PxByPyByPz:
                    for m in range(self.nOrbitals):
                        gp = cell.gridpoints[i,j,k]
#                         gp.phi[m] = np.sin(gp.x)/(abs(gp.x)+abs(gp.y)+abs(gp.z))/(m+1)
                        gp.phi[m] = np.random.rand(1)
        
    def initializeDensityFromAtomicData(self):
        for _,cell in self.masterList:
            if cell.leaf==True:
                for i,j,k in self.PxByPyByPz:
                    gp = cell.gridpoints[i,j,k]
                    gp.rho = 0.0
                    for atom in self.atoms:
                        r = np.sqrt( (gp.x-atom.x)**2 + (gp.y-atom.y)**2 + (gp.z-atom.z)**2 )
                        try:
                            gp.rho += atom.interpolators['density'](r)
                        except ValueError:
                            gp.rho += 0.0   # if outside the interpolation range, assume 0.
                
#                 if hasattr(cell, 'densityPoints'):
#                     for i,j,k in cell.PxByPyByPz_density:
#                         dp = cell.densityPoints[i,j,k]
#                         dp.rho = 0.0
#                         for atom in self.atoms:
#                             r = np.sqrt( (dp.x-atom.x)**2 + (dp.y-atom.y)**2 + (dp.z-atom.z)**2 )
#                             try:
#                                 dp.rho += atom.interpolators['density'](r)
#                             except ValueError:
#                                 dp.rho += 0.0   # if outside the interpolation range, assume 0.
#         
# #         self.normalizeDensity()
#         self.integrateDensityBothMeshes()
                            
                        
    
    def initializeOrbitalsFromAtomicData(self,onlyFillOne=False):
        
        aufbauList = ['10',                                     # n+ell = 1
                      '20',                                     # n+ell = 2
                      '21', '30',                               # n+ell = 3
                      '31', '40', 
                      '32', '41', '50'
                      '42', '51', '60'
                      '43', '52', '61', '70']

        timer = Timer()
        timer.start()
        orbitalIndex=0
        
        
#         print('Hard coding nAtomicOrbitals to 2 for the oxygen atom.')
# #         print('Hard coding nAtomicOrbitals to 0 for the second hydrogen atom.')
#         self.atoms[1].nAtomicOrbitals = 2
#         self.atoms[1].nAtomicOrbitals = 0
    
        for atom in self.atoms:
            if onlyFillOne == True:
                print('Setting number of orbitals equal to 1 for oxygen, just for testing the deep state without initializing everything')
                nAtomicOrbitals = 1
            else:
                nAtomicOrbitals = atom.nAtomicOrbitals
            
            print('Initializing orbitals for atom Z = %i located at (x, y, z) = (%6.3f, %6.3f, %6.3f)' 
                      %(atom.atomicNumber, atom.x,atom.y,atom.z))
            print('Orbital index = %i'%orbitalIndex)            
            singleAtomOrbitalCount=0
            for nell in aufbauList:
                
                if singleAtomOrbitalCount< nAtomicOrbitals:  
                    n = int(nell[0])
                    ell = int(nell[1])
                    psiID = 'psi'+str(n)+str(ell)
#                     print('Using ', psiID)
                    for m in range(-ell,ell+1):
                        for _,cell in self.masterList:
                            if cell.leaf==True:
                                for i,j,k in self.PxByPyByPz:
                                    gp = cell.gridpoints[i,j,k]
                                    dx = gp.x-atom.x
                                    dy = gp.y-atom.y
                                    dz = gp.z-atom.z
                                    r = np.sqrt( dx**2 + dy**2 + dz**2 )
                                    inclination = np.arccos(dz/r)
                                    azimuthal = np.arctan2(dy,dx)
                                    
                                
    #                                     Y = sph_harm(m,ell,azimuthal,inclination)*np.exp(-1j*m*azimuthal)
                                    if m<0:
                                        Y = (sph_harm(m,ell,azimuthal,inclination) + (-1)**m * sph_harm(-m,ell,azimuthal,inclination))/np.sqrt(2) 
                                    if m>0:
                                        Y = 1j*(sph_harm(m,ell,azimuthal,inclination) - (-1)**m * sph_harm(-m,ell,azimuthal,inclination))/np.sqrt(2)
#                                     if ( (m==0) and (ell>1) ):
                                    if ( m==0 ):
                                        Y = sph_harm(m,ell,azimuthal,inclination)
#                                     if ( (m==0) and (ell<=1) ):
#                                         Y = 1
                                    if abs(np.imag(Y)) > 1e-14:
                                        print('imag(Y) ', np.imag(Y))
    #                                     Y = np.real(sph_harm(m,ell,azimuthal,inclination))
                                    try:
                                        gp.phi[orbitalIndex] = atom.interpolators[psiID](r)*np.real(Y)
                                    except ValueError:
                                        gp.phi[orbitalIndex] = 0.0
                                        
                        
                        
                        print('Orbital %i filled with (n,ell,m) = (%i,%i,%i) ' %(orbitalIndex,n,ell,m))
                        orbitalIndex += 1
                        singleAtomOrbitalCount += 1
                    
#                 else:
#                     n = int(nell[0])
#                     ell = int(nell[1])
#                     psiID = 'psi'+str(n)+str(ell)
#                     print('Not using ', psiID)
                        
        if orbitalIndex < self.nOrbitals:
            print("Didn't fill all the orbitals.  Should you initialize more?  Randomly, or using more single atom data?")
        if orbitalIndex > self.nOrbitals:
            print("Filled too many orbitals, somehow.  That should have thrown an error and never reached this point.")
                        

        for m in range(self.nOrbitals):
            self.normalizeOrbital(m)
    

        
    def initializeForBerylliumAtom(self):
        print('Initializing orbitals for beryllium atom exclusively. ')

        # Generalized for any atoms.  Not complete yet.  
        timer = Timer()
        timer.start()
        
        interpolators = np.empty(self.nOrbitals,dtype=object)
#         count=0
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
                if r >= 29:
                    gp.setPhi(0,0)
                    gp.setPhi(0,1)
                else:
                    gp.setPhi(interpolators[0](r),0)
                    gp.setPhi(interpolators[1](r),1)
                    
                    #  Perturb the initial wavefunction
                
                    gp.setPhi( gp.phi[0] + np.sin(r),0)
                    gp.setPhi( gp.phi[1] + np.sin(2*r),1)
                   
        timer.stop()
        print('Using perturbed versions of single-atom data, since this is a single atom calculation.')
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
                    if r > 29:
                        gp.setPhi(gp.phi[0] + 0,0)
                    else:
                        gp.setPhi(gp.phi[0] + interpolators[0](r),0)
                   
        timer.stop()
        print('Initialization from single atom data took %f.3 seconds.' %timer.elapsedTime)
        
        
    def buildTree(self,minLevels,maxLevels, divideCriterion, divideParameter, initializationType='atomic',printNumberOfCells=False, printTreeProperties = True, onlyFillOne=False): # call the recursive divison on the root of the tree
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
                            maxDepthAchieved, minDepthAchieved, levelCounter = recursiveDivide(self,Cell.children[i,j,k], 
                                                                                minLevels, maxLevels, divideCriterion, divideParameter, 
                                                                                levelCounter, printNumberOfCells, maxDepthAchieved, 
                                                                                minDepthAchieved, currentLevel+1)
            
            elif currentLevel < maxLevels:
                
                if currentLevel < minLevels:
                    Cell.divideFlag = True 
#                     print('dividing cell ', Cell.uniqueID, ' because it is below the minimum level')
                else:  
                    if ( (divideCriterion == 'LW1') or (divideCriterion == 'LW2') or (divideCriterion == 'LW3') or (divideCriterion == 'LW3_modified') or 
                         (divideCriterion == 'LW4') or (divideCriterion == 'LW5') or(divideCriterion == 'Phani') ):
#                         print('checking divide criterion for cell ', Cell.uniqueID)
                        Cell.checkIfAboveMeshDensity(divideParameter,divideCriterion)  
                    else:                        
                        Cell.checkIfCellShouldDivide(divideParameter)
                    
                if Cell.divideFlag == True:
                    xdiv = (Cell.xmax + Cell.xmin)/2   
                    ydiv = (Cell.ymax + Cell.ymin)/2   
                    zdiv = (Cell.zmax + Cell.zmin)/2   
                    Cell.divide(xdiv, ydiv, zdiv, printNumberOfCells)

                    for i,j,k in TwoByTwoByTwo:
                        maxDepthAchieved, minDepthAchieved, levelCounter = recursiveDivide(self,Cell.children[i,j,k], minLevels, maxLevels, divideCriterion, divideParameter, levelCounter, printNumberOfCells, maxDepthAchieved, minDepthAchieved, currentLevel+1)
                else:
                    minDepthAchieved = min(minDepthAchieved, currentLevel)
                    
                    
            maxDepthAchieved = max(maxDepthAchieved, currentLevel)                                                                                                                                                       
            return maxDepthAchieved, minDepthAchieved, levelCounter
        
        timer.start()
        levelCounter=0
        self.maxDepthAchieved, self.minDepthAchieved, self.treeSize = recursiveDivide(self, self.root, minLevels, maxLevels, divideCriterion, divideParameter, levelCounter, printNumberOfCells, maxDepthAchieved=0, minDepthAchieved=maxLevels, currentLevel=0 )
        
#         refineRadius = 0.01
#         print('Refining uniformly within radius ', refineRadius, ' which is set within the buildTree method.')
#         self.uniformlyRefineWithinRadius(refineRadius)
#         refineRadius /= 2
#         print('Refining uniformly within radius ', refineRadius, ' which is set within the buildTree method.')
#         self.uniformlyRefineWithinRadius(refineRadius)
        
        """ Count the number of unique leaf cells and gridpoints and set initial external potential """
        self.numberOfGridpoints = 0
        self.numberOfCells = 0
        closestToOrigin = 10
        for _,cell in self.masterList:
            if cell.leaf==True:
                self.numberOfCells += 1
                for i,j,k in self.PxByPyByPz:
                    if not hasattr(cell.gridpoints[i,j,k], "counted"):
                        self.numberOfGridpoints += 1
                        cell.gridpoints[i,j,k].counted = True
#                         cell.gridpoints[i,j,k].setExternalPotential(self.atoms, self.gaugeShift)
                        gp = cell.gridpoints[i,j,k]
                        r = np.sqrt( gp.x*gp.x + gp.y*gp.y + gp.z*gp.z )
                        if r < closestToOrigin:
                            closestToOrigin = np.copy(r)
                            closestCoords = [gp.x, gp.y, gp.z]
                            closestMidpoint = [cell.xmid, cell.ymid, cell.zmid]
        
        self.rmin = closestToOrigin
        
        
                        
        for _,cell in self.masterList:
            for i,j,k in self.PxByPyByPz:
                if hasattr(cell.gridpoints[i,j,k], "counted"):
                    cell.gridpoints[i,j,k].counted = None
         
        
        print('Number of gridpoints: ', self.numberOfGridpoints)

        self.computeDerivativeMatrices()
        self.initializeDensityFromAtomicData()
        ### INITIALIZE ORBTIALS AND DENSITY ####
        if initializationType=='atomic':
            if onlyFillOne == True:
                self.initializeOrbitalsFromAtomicData(onlyFillOne=True)
            else:
                self.initializeOrbitalsFromAtomicData()
        elif initializationType=='random':
            self.initializeOrbitalsRandomly()
#         self.orthonormalizeOrbitals()
            
        
#         self.normalizeDensity()
        
            
            
#         self.initializeForHydrogenMolecule()
#         self.initializeForBerylliumAtom()
#         self.orthonormalizeOrbitals()
        
        
        timer.stop()
                    
        if printTreeProperties == True: 
            print("Tree build completed. \n"
                  "Domain Size:                 [%.1f, %.1f] \n"
                  "Divide Criterion:             %s \n"
                  "Divide Parameter:            %1.2e \n"
                  "Total Number of Cells:       %i \n"
                  "Total Number of Leaf Cells:  %i \n"
                  "Total Number of Gridpoints:  %i \n"
                  "Minimum Depth                %i levels \n"
                  "Maximum Depth:               %i levels \n"
                  "Cell Order:                  %i \n"
                  "Construction time:           %.3g seconds."
                   
                  %(self.xmin, self.xmax, divideCriterion,divideParameter, self.treeSize, self.numberOfCells, self.numberOfGridpoints, self.minDepthAchieved,self.maxDepthAchieved, self.px, timer.elapsedTime))
            print('Closest gridpoint to origin: ', closestCoords)
            print('For a distance of: ', closestToOrigin)
            print('Part of a cell centered at: ', closestMidpoint) 


                 
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
                  "Gauge Shift:                 %f \n"
                  "Tolerance1:                  %1.2e \n"
                  "Tolerance2:                  %1.2e \n"
                  "Total Number of Cells:       %i \n"
                  "Total Number of Gridpoints:  %i \n"
                  "Minimum Depth                %i levels \n"
                  "Maximum Depth:               %i levels \n"
                  "Construction time:           %.3g seconds." 
                  %(self.xmin, self.xmax, self.gaugeShift, 
                    divideTolerance1, divideTolerance2, 
                    self.treeSize, self.numberOfGridpoints, 
                    self.minDepthAchieved,self.maxDepthAchieved, 
                    timer.elapsedTime))
    

    def uniformlyRefine(self):
        
        cellCounter = 0
        for _,cell in self.masterList:
            if cell.leaf==True:
                cellCounter += 1
                cell.divideFlag = True
        
        print('Uniformly refining all %i cells.' %cellCounter)
        
        for _,cell in self.masterList:
            if cell.leaf==True:
#                 print('Dividing cell ', cell.uniqueID)
                if  cell.divideFlag == True:
#                     print('Dividing cell ', cell.uniqueID)
                    cell.divide(cell.xmid, cell.ymid, cell.zmid)
                    for i,j,k in TwoByTwoByTwo:
                        cell.children[i,j,k].divideFlag = False
                    cell.divideFlag = False
                    
        cellCounter = 0
        self.numberOfGridpoints = 0
        self.numberOfCells = 0
        for _,cell in self.masterList:
            if cell.leaf==True:
                self.numberOfCells += 1
                self.numberOfGridpoints += self.px * self.py * self.pz
                
        print('Now there are %i cells and %i gridpoints.' %(self.numberOfCells, self.numberOfGridpoints) )
        
        self.computeDerivativeMatrices()

        ### INITIALIZE ORBTIALS AND DENSITY ####
        self.initializeDensityFromAtomicData()
        self.initializeOrbitalsFromAtomicData()            
#         self.normalizeDensity()
        
        self.maxDepthAchieved += 1
        self.minDepthAchieved += 1
        
    def uniformlyRefineWithinRadius(self,R):
        
        cellCounter = 0
        for _,cell in self.masterList:
            if cell.leaf==True:
                for atom in self.atoms:
                    r = np.sqrt( (cell.xmid - atom.x)**2 + (cell.ymid - atom.y)**2 + (cell.zmid - atom.z)**2)
                    if r < R:
                        cellCounter += 1
                        cell.divideFlag = True
        
        print('Uniformly refining %i cells within radius %1.2e.' %(cellCounter,R))
        
        for _,cell in self.masterList:
            if cell.leaf==True:
#                 print('Dividing cell ', cell.uniqueID)
                if  cell.divideFlag == True:
#                     print('Dividing cell ', cell.uniqueID)
                    cell.divide(cell.xmid, cell.ymid, cell.zmid)
                    for i,j,k in TwoByTwoByTwo:
                        cell.children[i,j,k].divideFlag = False
                    cell.divideFlag = False
                    
        cellCounter = 0
        self.numberOfGridpoints = 0
        self.numberOfCells = 0
        for _,cell in self.masterList:
            if cell.leaf==True:
                self.maxDepthAchieved = max(self.maxDepthAchieved, cell.level)
                self.numberOfCells += 1
                self.numberOfGridpoints += self.px * self.py * self.pz
                
        print('Now there are %i cells and %i gridpoints.' %(self.numberOfCells, self.numberOfGridpoints) )
        print('Maximum depth ', self.maxDepthAchieved)
        
        self.computeDerivativeMatrices()

        ### INITIALIZE ORBTIALS AND DENSITY ####
#         self.initializeOrbitalsFromAtomicData()            
#         self.initializeDensityFromAtomicData()
#         self.normalizeDensity()
        
        self.maxDepthAchieved += 1
        self.minDepthAchieved += 1
        
        
    
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
        
        def CellupdateVxcAndVeff(cell,exchangeFunctional, correlationFunctional):
            '''
            After density is updated the convolution gets called to update V_coulomb.
            Now I need to update v_xc, then get the new value of v_eff. 
            '''
            
            rho = np.empty((cell.px,cell.py,cell.pz))
            
            for i,j,k in cell.PxByPyByPz:
                rho[i,j,k] = cell.gridpoints[i,j,k].rho
                
            exchangeOutput = exchangeFunctional.compute(rho)
            correlationOutput = correlationFunctional.compute(rho)
            
            epsilon_exchange = np.reshape(exchangeOutput['zk'],np.shape(rho))
            epsilon_correlation = np.reshape(correlationOutput['zk'],np.shape(rho))
            
            VRHO_exchange = np.reshape(exchangeOutput['vrho'],np.shape(rho))
            VRHO_correlation = np.reshape(correlationOutput['vrho'],np.shape(rho))
            
            for i,j,k in cell.PxByPyByPz:
                cell.gridpoints[i,j,k].epsilon_x = epsilon_exchange[i,j,k]
                cell.gridpoints[i,j,k].epsilon_c = epsilon_correlation[i,j,k]
                cell.gridpoints[i,j,k].v_x = VRHO_exchange[i,j,k]
                cell.gridpoints[i,j,k].v_c = VRHO_correlation[i,j,k]
#                 cell.gridpoints[i,j,k].v_xc = 0
                cell.gridpoints[i,j,k].updateVeff()
            
        for _,cell in self.masterList:
            if cell.leaf == True:
                CellupdateVxcAndVeff(cell,self.exchangeFunctional, self.correlationFunctional)

    def updateDensityAtQuadpoints(self, mixingScheme='Simple'):
        def CellUpdateDensity(cell,mixingScheme):
            for i,j,k in self.PxByPyByPz:
                newRho = 0
                for m in range(self.nOrbitals):
                    newRho += cell.tree.occupations[m] * cell.gridpoints[i,j,k].phi[m]**2
                if mixingScheme=='None':
                    cell.gridpoints[i,j,k].rho = newRho
                elif mixingScheme=='Simple':
                    cell.gridpoints[i,j,k].rho = ( self.mixingParameter*cell.gridpoints[i,j,k].rho + 
                        (1-self.mixingParameter)*newRho )
                else: 
                    print('Not a valid density mixing scheme.')
                    return
            
            
        for _,cell in self.masterList:
            if cell.leaf==True:
                CellUpdateDensity(cell,mixingScheme)
     
    def normalizeDensity(self):            
        def integrateDensity(cell):
            rho = np.empty((cell.px,cell.py,cell.pz))
                        
            for i,j,k in cell.PxByPyByPz:
                gp = cell.gridpoints[i,j,k]
                rho[i,j,k] = gp.rho
            
            return np.sum( cell.w * rho )
        
        def integrateDensity_secondaryMesh(cell):
            rho = np.empty((cell.pxd,cell.pyd,cell.pzd))
                        
            for i,j,k in cell.PxByPyByPz_density:
                dp = cell.densityPoints[i,j,k]
                rho[i,j,k] = dp.rho
            
            return np.sum( cell.w_density * rho )
        
        print('Normalizing density... are you sure?')
        A = 0.0
        B = 0.0
        for _,cell in self.masterList:
            if cell.leaf == True:
                A += integrateDensity(cell)
                B += integrateDensity_secondaryMesh(cell)
        if B==0.0:
            print('Warning, integrated density to 0')
#         print('Raw computed density ', B)
        for _,cell in self.masterList:
            if cell.leaf == True:
                for i,j,k in cell.PxByPyByPz:
                    cell.gridpoints[i,j,k].rho/=(A/self.nElectrons)
                for i,j,k in cell.PxByPyByPz_density:
                    cell.densityPoints[i,j,k].rho/=(B/self.nElectrons)
                    
#         B = 0.0
#         for id,cell in self.masterList:
#             if cell.leaf == True:
#                 B += integrateDensity(cell)
#         print('Integrated density after normalization ', B)

    def integrateDensityBothMeshes(self):            
        def integrateDensity(cell):
            rho = np.empty((cell.px,cell.py,cell.pz))
                        
            for i,j,k in cell.PxByPyByPz:
                gp = cell.gridpoints[i,j,k]
                rho[i,j,k] = gp.rho
            
            return np.sum( cell.w * rho )
        
        def integrateDensity_secondaryMesh(cell):
            rho = np.empty((cell.pxd,cell.pyd,cell.pzd))
                        
            for i,j,k in cell.PxByPyByPz_density:
                dp = cell.densityPoints[i,j,k]
                rho[i,j,k] = dp.rho
            
            return np.sum( cell.w_density * rho )
        A = 0.0
        B = 0.0
        C = 0.0
        for _,cell in self.masterList:
            if cell.leaf == True:
                A += integrateDensity(cell)
                B += integrateDensity_secondaryMesh(cell)
                
                C += (integrateDensity(cell) - integrateDensity_secondaryMesh(cell))**2 * cell.volume
        
        C = np.sqrt(C)

        print('Original mesh computed density  ', A)
        print('Secondary mesh computed density ', B)
        print('L2 norm of the difference (cell averaged) : ', C)

    
                
            
    
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
        
        V_x = 0.0
        V_c = 0.0
        V_coulomb = 0.0
        E_x = 0.0
        E_c = 0.0
        E_electronNucleus = 0.0
        
        
        for _,cell in self.masterList:
            if cell.leaf == True:
                V_x += integrateCellDensityAgainst__(cell,'v_x')
                V_c += integrateCellDensityAgainst__(cell,'v_c')
                V_coulomb += integrateCellDensityAgainst__(cell,'v_coulomb')
                E_x += integrateCellDensityAgainst__(cell,'epsilon_x')
                E_c += integrateCellDensityAgainst__(cell,'epsilon_c')
                E_electronNucleus += integrateCellDensityAgainst__(cell,'v_ext')
                
        self.totalVx = V_x
        self.totalVc = V_c
        self.totalVcoulomb = V_coulomb
#         self.totalElectrostatic = -1/2*V_coulomb - V_x - V_c + self.nuclearNuclear + self.totalOrbitalPotential
        self.totalElectrostatic = 1/2*V_coulomb + self.nuclearNuclear + E_electronNucleus
        self.totalEx = E_x
        self.totalEc = E_c
        self.totalVext = E_electronNucleus
#         print('Total V_xc : ')
        
        print('Electrostatic Energies:')
        print('Hartree:         ', 1/2*V_coulomb)
        print('External:        ', E_electronNucleus)
        print('Nuclear-Nuclear: ', self.nuclearNuclear)
#         print('Sanity check...')
#         print('Band minus kinetic: ', self.totalBandEnergy - self.totalKinetic)
#         print('Electrostatic minus external and Nuclear plus V_x and V_c: ', self.totalElectrostatic - self.nuclearNuclear - E_electronNucleus + self.totalVc + self.totalVx)
        
#         self.totalPotential = -1/2*V_coulomb + E_xc - V_xc 
#         self.totalPotential = -1/2*V_coulomb + E_x + E_c - V_x - V_c + self.nuclearNuclear
        self.totalPotential = self.totalElectrostatic +  E_x + E_c # - V_x - V_c 
                
        
       
    def computeBandEnergy(self):
        # sum over the kinetic energies of all orbitals
        self.totalBandEnergy = 0.0
        for i in range(self.nOrbitals):
            self.totalBandEnergy += self.occupations[i]*(self.orbitalEnergies[i] - self.gaugeShift)  # +1 due to the gauge potential
        
    
    
    def updateTotalEnergy(self,gradientFree):
        self.computeBandEnergy()
        self.computeTotalPotential()
        self.totalKinetic = self.totalBandEnergy - self.totalVcoulomb - self.totalVx - self.totalVc - self.totalVext
        
        if gradientFree==True:
            self.E = self.totalBandEnergy - 1/2 * self.totalVcoulomb + self.totalEx + self.totalEc - self.totalVx - self.totalVc + self.nuclearNuclear
            print('Updating total energy without explicit kinetic evaluation.')
        elif gradientFree==False:
            print('Updating total energy WITH explicit kinetic evaluation.')
            self.E = self.totalKinetic + self.totalPotential
        else:
            print('Invalid option for gradientFree.')
            print('gradientFree = ', gradientFree)
            print('type: ', type(gradientFree))
            return
        

    
    def computeOrbitalPotentials(self,targetEnergy=None, saveAsReference=False): 
        
        self.orbitalPotential = np.zeros(self.nOrbitals)  
        for _,cell in self.masterList:
            if cell.leaf == True:
                cell.computeOrbitalPotentials(targetEnergy)
                self.orbitalPotential += cell.orbitalPE
                if saveAsReference == True:
                    cell.referencePotential = np.copy(cell.orbitalPE[0])
                
        self.totalOrbitalPotential = np.sum( (self.orbitalPotential - self.gaugeShift) * self.occupations)
                       
    def computeOrbitalKinetics(self,targetEnergy=None, saveAsReference=False):
        print('Computing orbital kinetics using Gradients')
        self.orbitalKinetic = np.zeros(self.nOrbitals)
        for _,cell in self.masterList:
            if cell.leaf == True:
                cell.computeOrbitalKinetics(targetEnergy)
                self.orbitalKinetic += cell.orbitalKE
                if saveAsReference == True:
                    cell.referenceKinetic = np.copy(cell.orbitalKE[0])
        
        self.totalKinetic = np.sum(self.occupations*self.orbitalKinetic)
            
        
    def scrambleOrbital(self,m):
        # randomize orbital because its energy went > Vgauge
        for _,cell in self.masterList:
            if cell.leaf==True:
                val = np.random.rand(1)
                for i,j,k in self.PxByPyByPz:
                    gp = cell.gridpoints[i,j,k]
                    r = np.sqrt(gp.x*gp.x + gp.y*gp.y + gp.z*gp.z)
#                     gp.phi[m] = val/r
                    gp.phi[m] = val
    
    def softenOrbital(self,m):
        print('Softening orbital ', m)
        # randomize orbital because its energy went > Vgauge
        for _,cell in self.masterList:
            if cell.leaf==True:
                for i,j,k in self.PxByPyByPz:
                    gp = cell.gridpoints[i,j,k]
                    r = np.sqrt(gp.x*gp.x + gp.y*gp.y + gp.z*gp.z)
                    gp.phi[m] *= np.exp(-r)
                    
    def zeroOutOrbital(self,m):
#         print('Zeroing orbital ', m)
        print('setting orbital %i to exp(-r).'%m)
        # randomize orbital because its energy went > Vgauge
        for _,cell in self.masterList:
            if cell.leaf==True:
                for i,j,k in self.PxByPyByPz:
                    gp = cell.gridpoints[i,j,k]
                    r = np.sqrt(gp.x*gp.x + gp.y*gp.y + gp.z*gp.z)
                    gp.phi[m] = np.exp(-r)
        self.normalizeOrbital(m)
        
    def resetOrbitalij(self,m,n):
        # set orbital m equal to orbital n
        for _,cell in self.masterList:
            if cell.leaf==True:
                for i,j,k in self.PxByPyByPz:
                    gp = cell.gridpoints[i,j,k]
                    gp.phi[m] = gp.phi[n]
                    
    def compareToReferenceEnergies(self, refineFraction = 0.0):
        
        cellErrorsList = []
        for _,cell in self.masterList:
            if cell.leaf==True:
                kineticError = abs(cell.orbitalKE[0] - cell.referenceKinetic)
                potentialError = abs(cell.orbitalPE[0] - cell.referencePotential)
#                 print(kineticError)
#                 print(potentialError)
                totalError = potentialError + kineticError
                radius = np.sqrt(cell.xmid**2 + cell.ymid**2 + cell.zmid**2)
                cellErrorsList.append( [cell.level, radius, float(kineticError), float(potentialError), float(totalError), cell ] )
        
#         sortedByKinetic = sorted(cellErrorsList,key=lambda x:x[2], reverse=True)
        sortedByPotential = sorted(cellErrorsList,key=lambda x:x[3], reverse=True)
        sortedByTotal = sorted(cellErrorsList,key=lambda x:x[4], reverse=True)
        
#         print(cellErrorsList)
#         print('Twenty worst cells for kinetic: ',sortedByKinetic[:20])
#         print('Twenty worst cells for potential: ',sortedByPotential[:20])
        
        if refineFraction > 0:
            numToDivide = int(np.ceil(len(sortedByTotal)*refineFraction))
            print('Refining worst ', numToDivide, 'cells in terms of deepest orbital energy errors. ' )
            for i in range(numToDivide):
#                 cell = sortedByTotal[i][5]
                cell = sortedByPotential[i][5]
                cell.divide(cell.xmid,cell.ymid,cell.zmid)
        
        
        return
    
    
    def updateOrbitalEnergies(self,newOccupations=True,correctPositiveEnergies=True,sortByEnergy=True,targetEnergy=None, saveAsReference=False):
#         print()
        start = time.time()
        self.computeOrbitalKinetics(targetEnergy, saveAsReference)
        kinTime = time.time()-start
        start=time.time()
        self.computeOrbitalPotentials(targetEnergy, saveAsReference)
        potTime = time.time()-start
        self.orbitalEnergies = self.orbitalKinetic + self.orbitalPotential
#         print('Orbital Kinetic Energy:   ', self.orbitalKinetic)
#         print('Orbital Potential Energy: ', self.orbitalPotential)
#         print('Orbital Energy:           ', self.orbitalEnergies)
        ### CHECK IF NEED TO RE-ORDER ORBITALS ###
        if sortByEnergy==True:
            if not np.all(self.orbitalEnergies[:-1] <= self.orbitalEnergies[1:]):
                print('Need to re-order orbitals.')
#                 print('Orbital Energies before sorting: ', self.orbitalEnergies)
                self.sortOrbitalsAndEnergies()
                self.updateOrbitalEnergies()
#             print('After sorting...')
            else:
                print('Orbital Energy:           ', self.orbitalEnergies)
                    
        #         print('Kinetic took %2.3f, Potential took %2.3f seconds' %(kinTime,potTime))
                energyResetFlag = 0
                if correctPositiveEnergies==True:
                    for m in range(self.nOrbitals):
        #                 if self.orbitalEnergies[m] > self.gaugeShift:
        #                     if m==0:
        #                         print('phi0 energy > gauge shift, setting to gauge shift - 3')
        #                         self.orbitalEnergies[m] = self.gaugeShift-3
        #                     else:
        #                         print('orbital %i energy > gauge shift.  Setting orbital to same as %i, energy slightly higher' %(m,m-1))
        #                         self.resetOrbitalij(m,m-1)
        #                         self.orbitalEnergies[m] = self.orbitalEnergies[m-1] + 0.1
                        if self.orbitalEnergies[m] > 0.0:
        #                 if self.orbitalEnergies[m] > self.gaugeShift:
                            print('Warning: %i orbital energy > 0.  Resetting to gauge shift/2.' %m)
        #                     print('Warning: %i orbital energy > gauge shift.  Resetting to gauge shift.' %m)
                            self.orbitalEnergies[m] = self.gaugeShift/2
        #                     print('Warning: %i orbital energy > gaugeShift. Setting phi to zero' %m)
                            
        #                     self.zeroOutOrbital(m)
        #                     self.orbitalEnergies[m] = self.gaugeShift - 1/(m+1)
        #                     print('Setting energy to %1.3e' %self.orbitalEnergies[m])
        #                 self.scrambleOrbital(m)
        #                 self.softenOrbital(m)
        #                     energyResetFlag=1
                    self.sortOrbitalsAndEnergies()
                
                
        #         if energyResetFlag==1:
        #             print('Re-orthonormalizing orbitals after scrambling those with positive energy.')
        #             print('Re-orthonormalizing orbitals after scrambling those with positive energy.')
        #             self.orthonormalizeOrbitals()
        #             self.updateOrbitalEnergies()
        
                
        else: 
            print('Orbital Energy:           ', self.orbitalEnergies)
#                 print()
        if newOccupations==True:
            self.computeOccupations()
#             print('Occupations: ', self.occupations)

    def updateOrbitalEnergies_NoGradients(self,targetEnergy,newOccupations=True):
        
        deltaE = 0.0
        normSqOfPsiNew = 0.0
        for _,cell in self.masterList:
            if cell.leaf==True:
                phi = np.zeros((cell.px,cell.py,cell.pz))
                phiNew = np.zeros((cell.px,cell.py,cell.pz))
                potential = np.zeros((cell.px,cell.py,cell.pz))
                for i,j,k in cell.PxByPyByPz:
                    gp = cell.gridpoints[i,j,k]
                    phi[i,j,k] = gp.phi[targetEnergy]
                    phiNew[i,j,k] = gp.phiNew
                    potential[i,j,k] = gp.v_eff
                
                deltaE -= np.sum( phi*potential*(phi-phiNew)*cell.w ) 
                normSqOfPsiNew += np.sum( phiNew**2 * cell.w)
#         deltaE /= np.sqrt(normSqOfPsiNew)
        deltaE /= (normSqOfPsiNew)
        print('Norm of psiNew = ', np.sqrt(normSqOfPsiNew))
        
#         print('Previous orbital energy: ', self.orbitalEnergies[targetEnergy])
        self.orbitalEnergies[targetEnergy] += deltaE
#         print('Updated orbital energy:  ', self.orbitalEnergies[targetEnergy])
#         print('Orbital Kinetic Energy:   ', self.orbitalKinetic)
#         print('Orbital Potential Energy: ', self.orbitalPotential)
#         print('Orbital Energy:           ', self.orbitalEnergies)
        ### CHECK IF NEED TO RE-ORDER ORBITALS ###
        
        if newOccupations==True:
            self.computeOccupations()
#             print('Occupations: ', self.occupations)

    def sortOrbitalsAndEnergies(self):
        newOrder = np.argsort(self.orbitalEnergies)
        print('New order: ', newOrder)
        for _,cell in self.masterList:
            if cell.leaf==True:
                for i,j,k in self.PxByPyByPz:
                    gp = cell.gridpoints[i,j,k]
                    
                    gp.sortOrbitals(newOrder)
                    
                    ###
#                     phiSorted = np.zeros_like(self.orbitalEnergies)
#                     for m in range(self.nOrbitals):
#                         phiSorted[m] = copy.deepcopy(gp.phi[m])
# #                     print('Pre-sort')
# #                     print(phiSorted)
#                     phiSorted = phiSorted[newOrder]
# #                     print('Post-sort')
# #                     print(phiSorted)
#                     
#                     for m in range(self.nOrbitals):
#                         gp.setPhi(copy.deepcopy(phiSorted[m]),m)
                    ### 
                        
#                     gp.phi = copy.deepcopy(phiSorted)
#                     for m in range(self.nOrbitals):
#                         gp.phi[m] = phiSorted[m]
#                     gp.phi = phiNew
        
#         newOccupations = np.zeros_like(self.occupations)
#         newKinetics = np.zeros_like(self.orbitalKinetic)
#         newPotentials = np.zeros_like(self.orbitalPotential)
#         newEnergies = np.zeros_like(self.orbitalEnergies)
#         for m in range(self.nOrbitals):
#             newOccupations[m] = self.occupations[newOrder[m]]
#             newKinetics[m] = self.orbitalKinetic[newOrder[m]]
#             newPotentials[m] = self.orbitalPotential[newOrder[m]]
#             newEnergies[m] = self.orbitalEnergies[newOrder[m]]
#         self.occupations = np.copy(newOccupations)
#         self.orbitalKinetic = np.copy(newKinetics)
#         self.orbitalPotential = np.copy(newPotentials)
#         self.orbitalEnergies = np.copy(newEnergies)
        
    
    def computeDerivativeMatrices(self):
        for _,cell in self.masterList:
            if cell.leaf==True:
                cell.computeDerivativeMatrices()
        
                    
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
        
    
    def computeWavefunctionResidual(self,energyLevel):
        energyEigenvalue = self.orbitalEnergies[energyLevel]
        kinetic = self.orbitalKinetic[energyLevel]
        potential = self.orbitalPotential[energyLevel]
        print('Energy Eigenvalue: ', energyEigenvalue)
#         print('kinetic part:      ', kinetic)
#         print('potential part:    ', potential)
        needToPrint=False
        
        L2residual = 0.0
        kineticResidual = 0.0
        potentialResidual = 0.0
        for _,cell in self.masterList:
            if cell.leaf==True:
#                 if not hasattr(cell, 'laplacian'):
#                     cell.computeLaplacian()
                phi = np.zeros((cell.px,cell.py,cell.pz))
                VeffPhi = np.zeros((cell.px,cell.py,cell.pz))
                
                for i,j,k in cell.PxByPyByPz:
                    gp = cell.gridpoints[i,j,k]
                    phi[i,j,k] = gp.phi[energyLevel]
                    VeffPhi[i,j,k] = gp.v_eff*gp.phi[energyLevel]
                
                laplacianPhi = ChebLaplacian3D(cell.DopenX, cell.DopenY, cell.DopenZ, cell.px, phi)
                Hphi = -1/2*laplacianPhi + VeffPhi
                
                L2residual += np.sum( (Hphi - energyEigenvalue*phi)**2 * cell.w )
                kineticResidual += np.sum( (-1/2*laplacianPhi - kinetic*phi)**2 * cell.w )
                potentialResidual += np.sum( (VeffPhi - potential*phi)**2 * cell.w )

                if ((needToPrint==True) and (cell.level>8) ):
                    print(energyEigenvalue*phi)
                    print()
                    print(-1/2*laplacianPhi)
                    print()
                    print(VeffPhi)
                    print()
                    print(Hphi - energyEigenvalue*phi)
                    
                    needToPrint=False
#                     return
        L2residual = np.sqrt( L2residual )
        
        print('L2 norm of wavefunction residual (H*psi-lambda*psi):   ', L2residual)
#         print('Kinetic portion:                                       ', kineticResidual)
#         print('Potential portion:                                     ', potentialResidual)
                
                
#             self.orbitalKE[targetEnergy] = 1/2*np.sum( self.w * gradPhiSq )
                    
                
        return
        
    def normalizeOrbital(self, n):
        """ Enforce integral phi*2 dxdydz == 1 for the nth orbital"""
        A = 0.0
        
        for _,cell in self.masterList:
            if cell.leaf == True:
                for i,j,k in self.PxByPyByPz:
#                     if abs(element[1].gridpoints[i,j,k].phi) > maxPhi:
#                         maxPhi = abs(element[1].gridpoints[i,j,k].phi)
#                     maxPhi = max( maxPhi, abs(element[1].gridpoints[i,j,k].phi))
                    A += cell.gridpoints[i,j,k].phi[n]**2*cell.w[i,j,k]
        
        if A<0.0:
            print('Warning: normalization value A is less than zero...')
        if A==0.0:
            print('Warning: normalization value A is zero...')

        
        """ Rescale wavefunction values, flip the flag """
        for _,cell in self.masterList:
            if cell.leaf==True:
                for i,j,k in self.PxByPyByPz:
                        cell.gridpoints[i,j,k].phi[n] /= np.sqrt(A)
#                     if element[1].gridpoints[i,j,k].normalized == False:
#                         element[1].gridpoints[i,j,k].phi /= np.sqrt(A)
#                         element[1].gridpoints[i,j,k].normalized = True
#                         maxPhi = max(maxPhi,abs(element[1].gridpoints[i,j,k].phi))
           
         
#     def orthogonalizeWavefunction(self,n):
#         """ Orthgononalizes phi against wavefunction n """
#         B = 0.0
#         for _,cell in self.masterList:
#             if cell.leaf == True:
#                 midpoint = cell.gridpoints[1,1,1]
#                 B += midpoint.phi*midpoint.finalWavefunction[n]*element[1].volume
#                 
#         """ Initialize the orthogonalization flag for each gridpoint """        
#         for _,cell in self.masterList:
#             if cell.leaf==True:
#                 for i,j,k in self.PxByPyByPz:
#                     cell.gridpoints[i,j,k].orthogonalized = False
#         
#         """ Subtract the projection, flip the flag """
#         for _,cell in self.masterList:
#             if cell.leaf==True:
#                 for i,j,k in self.PxByPyByPz:
#                     gridpoint = cell.gridpoints[i,j,k]
#                     if gridpoint.orthogonalized == False:
#                         gridpoint.phi -= B*gridpoint.finalWavefunction[n]
#                         gridpoint.orthogonalized = True
                        
    def orthonormalizeOrbitals(self, targetOrbital=None):
        
        def orthogonalizeOrbitals(tree,m,n):
            
#             print('Orthogonalizing orbital %i against %i' %(m,n))
            """ Compute the overlap, integral phi_r * phi_s """
            B = 0.0
            for _,cell in tree.masterList:
                if cell.leaf == True:
                    for i,j,k in self.PxByPyByPz:
                        phi_m = cell.gridpoints[i,j,k].phi[m]
                        phi_n = cell.gridpoints[i,j,k].phi[n]
                        B += phi_m*phi_n*cell.w[i,j,k]
#             print('Overlap before orthogonalization: ', B)

            """ Subtract the projection """
            for _,cell in tree.masterList:
                if cell.leaf==True:
                    for i,j,k in self.PxByPyByPz:
                        gridpoint = cell.gridpoints[i,j,k]
                        gridpoint.phi[m] -= B*gridpoint.phi[n]
            
            B = 0.0
            for _,cell in tree.masterList:
                if cell.leaf == True:
                    for i,j,k in self.PxByPyByPz:
                        phi_m = cell.gridpoints[i,j,k].phi[m]
                        phi_n = cell.gridpoints[i,j,k].phi[n]
                        B += phi_m*phi_n*cell.w[i,j,k]
#             print('Overlap after orthogonalization: ', B)
        
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
                            cell.gridpoints[i,j,k].phi[m] /= np.sqrt(A)
        
        if targetOrbital==None:
#         print('Orthonormalizing orbitals within tree structure up to orbital %i.' %maxOrbital)
            for m in range(self.nOrbitals):
                for n in range(m):
                    
                    orthogonalizeOrbitals(self,m,n)
                normalizeOrbital(self,m)
        else:
            for n in range(targetOrbital):
                orthogonalizeOrbitals(self,targetOrbital,n)
            normalizeOrbital(self,targetOrbital)
            
            
    
    """
    IMPORT/EXPORT FUNCTIONS
    """                   
#     def extractLeavesMidpointsOnly(self):
#         '''
#         Extract the leaves as a Nx4 array [ [x1,y1,z1,psi1], [x2,y2,z2,psi2], ... ]
#         '''
# #         leaves = np.empty((self.numberOfGridpoints,4))
#         leaves = []
#         counter=0
#         for element in self.masterList:
#             if element[1].leaf == True:
#                 midpoint =  element[1].gridpoints[1,1,1]
#                 leaves.append( [midpoint.x, midpoint.y, midpoint.z, midpoint.phi, potential(midpoint.x, midpoint.y, midpoint.z), element[1].volume ] )
#                 counter+=1 
#                 
#         print('Warning: extracting midpoints even tho this is a non-uniform mesh.')
#         return np.array(leaves)
    
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
                    leaves.append( [gridpt.x, gridpt.y, gridpt.z, gridpt.phi[orbitalNumber], gridpt.v_eff, cell.w[i,j,k], cell.volume ] )
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
                
        for _,cell in self.masterList:
            if cell.leaf == True:
                for i,j,k in cell.PxByPyByPz:
                    gridpt = cell.gridpoints[i,j,k]
                    leaves.append( [gridpt.x, gridpt.y, gridpt.z, gridpt.rho, cell.w[i,j,k] ] )
                            
        return np.array(leaves)
    
    def extractConvolutionIntegrand(self,containing=None): 
        '''
        Extract the leaves as a Nx5 array [ [x1,y1,z1,f1,w1], [x2,y2,z2,f2,w2], ... ] where f is the function being convolved
        '''
#         print('Extracting the gridpoints from all leaves...')
        leaves = []
                
        for _,cell in self.masterList:
            if cell.leaf == True:
                if (   (containing==None)  or   
                    (  
                           ( (cell.xmin<containing[0]) and (cell.xmax>containing[0]) )  and 
                           ( (cell.ymin<containing[1]) and (cell.ymax>containing[1]) )  and  
                           ( (cell.zmin<containing[2]) and (cell.zmax>containing[2]) ) )
                    ):
                    for i,j,k in cell.PxByPyByPz:
                        gridpt = cell.gridpoints[i,j,k]
                        leaves.append( [gridpt.x, gridpt.y, gridpt.z, gridpt.f, cell.w[i,j,k] ] )
                            
        return np.array(leaves)
    
    def extractGreenIterationIntegrand(self,m): 
        '''
        Extract the leaves as a Nx5 array [ [x1,y1,z1,f1,w1], [x2,y2,z2,f2,w2], ... ] where f is the function being convolved
        '''
#         print('Extracting the gridpoints from all leaves...')
        leaves = []
                
        for _,cell in self.masterList:
            if cell.leaf == True:
                for i,j,k in cell.PxByPyByPz:
                    gridpt = cell.gridpoints[i,j,k]
                    f = -2*gridpt.phi[m]*gridpt.v_eff
                    leaves.append( [gridpt.x, gridpt.y, gridpt.z, f, cell.w[i,j,k] ] )
                            
        return np.array(leaves)
    
    
    def computeSelfCellInterations(self,k):
        print("Computing interaction of each point with its own cell..")
        for _,cell in self.masterList:
            if cell.leaf == True:
                for i,j,k in cell.PxByPyByPz:
                    gp_t = cell.gridpoints[i,j,k]
                    
                    tempW = np.copy(cell.w)
                    tempW[i,j,k] = 0  # set the weight at the target point equal to zero
                    tempW *= cell.volume / np.sum(tempW)  # renormalize so that sum of weights = volume 
                    if abs(np.sum(tempW) - cell.volume)>1e-12:
                        print('Warning: temporary weights not summing to cell volume.')
                    # simple skipping    
                    gp_t.selfCellContribution = 0.0
                    
#                     # simple singularity subtraction
#                     gp_t.selfCellContribution = 4*pi*f_t/k**2
                    
                    for ii,jj,kk in cell.PxByPyByPz:
                        gp_s = cell.gridpoints[ii,jj,kk]
                        if gp_s != gp_t:
                            r = np.sqrt( (gp_t.x - gp_s.x)**2 +  (gp_t.y - gp_s.y)**2 + (gp_t.z - gp_s.z)**2   )
                            
                            # put whatever the integrand is:
                            
                            # simple skipping
#                             gp_t.selfCellContribution += tempW[ii,jj,kk] * gp_s.f *exp(-k*r)/(r)
                            
                            # simple singularity subtraction
                            gp_t.selfCellContribution += tempW[ii,jj,kk] * (gp_s.f - gp_t.f) *exp(-k*r)/(r)
        print('Done.')
        
        
    def computeSelfCellInterations_GaussianIntegralIdentity(self,containing=None):

        print("Computing interaction of each point with its own cell using Gaussian Integral Identity..")
        counter=0
        for _,cell in self.masterList:
            if cell.leaf == True:
                counter += 1
                if (   (containing==None)  or   
                    (  
                           ( (cell.xmin<containing[0]) and (cell.xmax>containing[0]) )  and 
                           ( (cell.ymin<containing[1]) and (cell.ymax>containing[1]) )  and  
                           ( (cell.zmin<containing[2]) and (cell.zmax>containing[2]) ) )
                    ):
                    if containing != None:
                        print('Computing self interaction for cell centered at ', cell.xmid, cell.ymid, cell.zmid, ' at a depth of ', cell.level)
                    
                    # determine an appropriate time discretization, based on the gridpoint spacing
                    maxDist = np.sqrt( (cell.xmax-cell.xmin)**2 + (cell.ymax-cell.ymin)**2 + (cell.zmax-cell.zmin)**2 )
                    gp1 = cell.gridpoints[0,0,0]
                    gp2 = cell.gridpoints[1,0,0]  # these should be tied for closest grid points within this cell
                    minDist = np.sqrt( (gp1.x-gp2.x)**2 + (gp1.y-gp2.y)**2 + (gp1.z-gp2.z)**2)
                    
                    tmax = 3/minDist   # this is picked to satisfy the closest points, which need the largest t
                    dt = (6/maxDist)/20              # this is picked to satisfy the farthet points, which need the smallest dt.  (40 intervals to cover first t=10/dist
                    
                    timeIntervals = int( np.ceil( tmax/dt ) )
    #                 timeIntervals = int( np.ceil(5*(maxDist/minDist)) )
#                     timeIntervals = 200
                    
#                     timeIntervals = int(np.ceil(tmax/0.5))
                    tvec = np.linspace(0,tmax,timeIntervals+1)
                    print('Cell ', counter, ' of ', self.numberOfCells)
                    print('Closest points: ', minDist)
                    print('Corner to corner: ',maxDist)
                    print('tmax = ', tmax)
                    print('dt = ', tvec[1]-tvec[0])
                    print('timeIntervals   = ', timeIntervals)
                    print()
                
                
                    # for each target point in cell...
#                     maxApproxError = 0
                    for i,j,k in cell.PxByPyByPz:
                        gp_t = cell.gridpoints[i,j,k]
                        gp_t.selfCellContribution = 0.0
                    
                        # integrate over time (midpoint or trapezoid)
                        for ell in range(timeIntervals+1):
#                             dt = tvec[ell+1]-tvec[ell]
#                             t = (tvec[ell+1]+tvec[ell])/2
                            dt = tvec[1]-tvec[0]
                            t = tvec[ell]
                            if ( (ell==0) or (ell==timeIntervals)):
                                dt /=2
                        
                            # integrate over space
                            for ii,jj,kk in cell.PxByPyByPz:
                                gp_s = cell.gridpoints[ii,jj,kk]
                                r = np.sqrt( (gp_t.x - gp_s.x)**2 +  (gp_t.y - gp_s.y)**2 + (gp_t.z - gp_s.z)**2   )
                                    
                                gp_t.selfCellContribution += cell.w[ii,jj,kk] * dt * gp_s.rho * np.exp(-t**2 * r**2)  
                                
                        gp_t.selfCellContribution *= 2/np.sqrt(np.pi)
                        
#                         gp_t.selfCellContribution += np.pi/tmax**2 * gp_t.rho  # corrction for truncation in t.  Should be good if tmax is large enough.
                
                else: # not the target cell we care about, set selfCellContribution equal to zero
                    for i,j,k in cell.PxByPyByPz:
                        gp_t = cell.gridpoints[i,j,k]
                        gp_t.selfCellContribution = 0.0 
        print('Done.')
        
    def computeSelfCellInterations_GaussianIntegralIdentity_singularitySubtraction(self,alpha,containing=None):

        print("Computing interaction of each point with its own cell using Gaussian Integral Identity..")
        counter=0
        for _,cell in self.masterList:
            if cell.leaf == True:
                counter += 1
                if (   (containing==None)  or   
                    (  
                           ( (cell.xmin<containing[0]) and (cell.xmax>containing[0]) )  and 
                           ( (cell.ymin<containing[1]) and (cell.ymax>containing[1]) )  and  
                           ( (cell.zmin<containing[2]) and (cell.zmax>containing[2]) ) )
                    ):
                    if containing != None:
                        print('Computing self interaction for cell centered at ', cell.xmid, cell.ymid, cell.zmid, ' at a depth of ', cell.level)
                    
                    # determine an appropriate time discretization, based on the gridpoint spacing
                    maxDist = np.sqrt( (cell.xmax-cell.xmin)**2 + (cell.ymax-cell.ymin)**2 + (cell.zmax-cell.zmin)**2 )
                    gp1 = cell.gridpoints[0,0,0]
                    gp2 = cell.gridpoints[1,0,0]  # these should be tied for closest grid points within this cell
                    minDist = np.sqrt( (gp1.x-gp2.x)**2 + (gp1.y-gp2.y)**2 + (gp1.z-gp2.z)**2)
                    
                    tmax = 5/minDist   # this is picked to satisfy the closest points, which need the largest t
                    dt = (6/maxDist)/5             # this is picked to satisfy the farthet points, which need the smallest dt.  (40 intervals to cover first t=10/dist
                    
                    timeIntervals = int( np.ceil( tmax/dt ) )
    #                 timeIntervals = int( np.ceil(5*(maxDist/minDist)) )
#                     timeIntervals = 200
                    
#                     timeIntervals = int(np.ceil(tmax/0.5))
                    tvec = np.linspace(0,tmax,timeIntervals+1)
                    print('Cell ', counter, ' of ', self.numberOfCells)
                    print('Closest points: ', minDist)
                    print('Corner to corner: ',maxDist)
                    print('tmax = ', tmax)
                    print('dt = ', tvec[1]-tvec[0])
                    print('timeIntervals   = ', timeIntervals)
                    print()
                
                
                    # for each target point in cell...
#                     maxApproxError = 0
                    for i,j,k in cell.PxByPyByPz:
                        gp_t = cell.gridpoints[i,j,k]
                        gp_t.selfCellContribution = 0.0
                    
                        # integrate over time (midpoint or trapezoid)
                        for ell in range(timeIntervals+1):
#                             dt = tvec[ell+1]-tvec[ell]
#                             t = (tvec[ell+1]+tvec[ell])/2
                            dt = tvec[1]-tvec[0]
                            t = tvec[ell]
                            if ( (ell==0) or (ell==timeIntervals)):
                                dt /=2
                        
                            # integrate over space
                            for ii,jj,kk in cell.PxByPyByPz:
                                gp_s = cell.gridpoints[ii,jj,kk]
                                r = np.sqrt( (gp_t.x - gp_s.x)**2 +  (gp_t.y - gp_s.y)**2 + (gp_t.z - gp_s.z)**2   )
                                    
                                gp_t.selfCellContribution += cell.w[ii,jj,kk] * dt * ( gp_s.rho - gp_t.rho*np.exp(-alpha**2*r**2) ) * np.exp(-t**2 * r**2)  
                                
                        gp_t.selfCellContribution *= 2/np.sqrt(np.pi)
                        
                        gp_t.selfCellContribution += np.pi/tmax**2 * gp_t.rho  # corrction for truncation in t.  Should be good if tmax is large enough.
                
                else: # not the target cell we care about, set selfCellContribution equal to zero
                    for i,j,k in cell.PxByPyByPz:
                        gp_t = cell.gridpoints[i,j,k]
                        gp_t.selfCellContribution = 0.0 
        print('Done.')
        
        
    def computeSelfCellInterations_GaussianIntegralIdentity_3intervals(self,t_lin, t_log,timeIntervals, containing=None):

        print("Computing interaction of each point with its own cell using Gaussian Integral Identity..")
        counter=0
        for _,cell in self.masterList:
            if cell.leaf == True:
                counter += 1
                if (   (containing==None)  or   
                    (  
                           ( (cell.xmin<containing[0]) and (cell.xmax>containing[0]) )  and 
                           ( (cell.ymin<containing[1]) and (cell.ymax>containing[1]) )  and  
                           ( (cell.zmin<containing[2]) and (cell.zmax>containing[2]) ) )
                    ):
                    if containing != None:
                        print('Computing self interaction for cell centered at ', cell.xmid, cell.ymid, cell.zmid, ' at a depth of ', cell.level)
                    
                    # determine an appropriate time discretization, based on the gridpoint spacing
                    maxDist = np.sqrt( (cell.xmax-cell.xmin)**2 + (cell.ymax-cell.ymin)**2 + (cell.zmax-cell.zmin)**2 )
                    gp1 = cell.gridpoints[0,0,0]
                    gp2 = cell.gridpoints[1,0,0]  # these should be tied for closest grid points within this cell
                    minDist = np.sqrt( (gp1.x-gp2.x)**2 + (gp1.y-gp2.y)**2 + (gp1.z-gp2.z)**2)
                    
#                     tmax = 3/minDist   # this is picked to satisfy the closest points, which need the largest t
#                     dt = (6/maxDist)/40              # this is picked to satisfy the farthet points, which need the smallest dt.  (40 intervals to cover first t=10/dist
                    
#                     timeIntervals = int( np.ceil( tmax/dt ) )
    #                 timeIntervals = int( np.ceil(5*(maxDist/minDist)) )
#                     timeIntervals = 200
                    
#                     timeIntervals = int(np.ceil(tmax/0.5))
                    tvec1 = np.linspace(0,t_lin,timeIntervals+1)
                    weights1 = tvec1[1:] - tvec1[:-1]
                    weights1[0] /= 2
                    weights1[-1] /= 2
                    tlogvec2 = np.linspace(np.log(t_lin), np.log(t_log), timeIntervals+1) 
                    tvec2 = np.exp(tlogvec2)
                    tvec2_mids = (tvec2[:-1] + tvec2[1:])/2
                    logweights2 = tlogvec2[1:] - tlogvec2[:-1]
                    weights2 = tvec2_mids*logweights2
                    weights2[0] /= 2
                    weights2[-1] /= 2
                    
                    tvec = np.append(tvec1,tvec2)
                    weights = np.append(weights1, weights2)
                    print('Cell ', counter, ' of ', self.numberOfCells)
                    print('Closest points: ', minDist)
                    print('Corner to corner: ',maxDist)
                    print('t_lin = ', t_lin)
                    print('t_log = ', t_log)
                    print('dt = ', tvec[1]-tvec[0])
                    print('timeIntervals   = ', timeIntervals)
                    print()
                
                
                    # for each target point in cell...
#                     maxApproxError = 0
                    for i,j,k in cell.PxByPyByPz:
                        gp_t = cell.gridpoints[i,j,k]
                        gp_t.selfCellContribution = 0.0
                    
                        # integrate over time (midpoint or trapezoid)
                        for ell in range(timeIntervals+1):
#                             dt = tvec[ell+1]-tvec[ell]
#                             t = (tvec[ell+1]+tvec[ell])/2
                            dt = weights[ell]
                            t = tvec[ell]
                        
                            # integrate over space
                            for ii,jj,kk in cell.PxByPyByPz:
                                gp_s = cell.gridpoints[ii,jj,kk]
                                r = np.sqrt( (gp_t.x - gp_s.x)**2 +  (gp_t.y - gp_s.y)**2 + (gp_t.z - gp_s.z)**2   )
                                    
                                gp_t.selfCellContribution += cell.w[ii,jj,kk] * dt * gp_s.rho * np.exp(-t**2 * r**2)  
                                
                        gp_t.selfCellContribution *= 2/np.sqrt(np.pi)
                        gp_t.selfCellContribution += np.pi/t_log**2 * gp_t.rho  # corrction for truncation in t.  Should be good if tmax is large enough.

                
                else: # not the target cell we care about, set selfCellContribution equal to zero
                    for i,j,k in cell.PxByPyByPz:
                        gp_t = cell.gridpoints[i,j,k]
                        gp_t.selfCellContribution = 0.0 
        print('Done.')
        
        
    def computeSelfCellInterations_GaussianIntegralIdentity_t_inner(self,containing=None):

        print("Computing interaction of each point with its own cell using Gaussian Integral Identity..")
        counter=0
        for _,cell in self.masterList:
            if cell.leaf == True:
                counter += 1
                if (   (containing==None)  or   
                    (  
                           ( (cell.xmin<containing[0]) and (cell.xmax>containing[0]) )  and 
                           ( (cell.ymin<containing[1]) and (cell.ymax>containing[1]) )  and  
                           ( (cell.zmin<containing[2]) and (cell.zmax>containing[2]) ) )
                    ):
                    if containing != None:
                        print('Computing self interaction for cell centered at ', cell.xmid, cell.ymid, cell.zmid)
                    
                    # determine an appropriate time discretization, based on the gridpoint spacing
#                     maxDist = np.sqrt( (cell.xmax-cell.xmin)**2 + (cell.ymax-cell.ymin)**2 + (cell.zmax-cell.zmin)**2 )
#                     gp1 = cell.gridpoints[0,0,0]
#                     gp2 = cell.gridpoints[1,1,1]  # these should be tied for closest grid points within this cell
#                     minDist = np.sqrt( (gp1.x-gp2.x)**2 + (gp1.y-gp2.y)**2 + (gp1.z-gp2.z)**2)
#                     
#                     tmax = 10/minDist
    #                 timeIntervals = int( np.ceil(5*(maxDist/minDist)) )
#                     timeIntervals = 200
                    
#                     timeIntervals = int(np.ceil(tmax/0.05))
#                     tvec = np.linspace(0,tmax,timeIntervals+1)
                    print('Cell ', counter, ' of ', self.numberOfCells)
#                     print('tmax = ', tmax)
#                     print('dt = ', tvec[1]-tvec[0])
#                     print('timeIntervals   = ', timeIntervals)
#                     print()
                
                
                    # for each target point in cell...
                    for i,j,k in cell.PxByPyByPz:
                        gp_t = cell.gridpoints[i,j,k]
                        gp_t.selfCellContribution = 0.0
                    
                    
                        
                        # integrate over space
                        for ii,jj,kk in cell.PxByPyByPz:
                            gp_s = cell.gridpoints[ii,jj,kk]
                            r = np.sqrt( (gp_t.x - gp_s.x)**2 +  (gp_t.y - gp_s.y)**2 + (gp_t.z - gp_s.z)**2   )
                            if r > 0.0:
                                tmax = int(np.ceil(5/r))
                                timeIntervals = 40
                                tvec = np.linspace(0,tmax,timeIntervals+1)
                                print('r = ', r)
                                print('tmax = ', tmax)
                                print('dt = ', tvec[1]-tvec[0])
                                print('timeIntervals   = ', timeIntervals)
                                print()
                                # integrate over time (midpoint or trapezoid)
                                for ell in range(timeIntervals):
                                    dt = tvec[ell+1]-tvec[ell]
                                    t = (tvec[ell+1]+tvec[ell])/2
                                    dt = tvec[ell+1]-tvec[ell]
                                    t = tvec[ell]
                                    
                                    
                                    gp_t.selfCellContribution += cell.w[ii,jj,kk] * dt * gp_s.rho * np.exp(-t**2 * r**2)  
                                
                        gp_t.selfCellContribution *= 2/np.sqrt(np.pi)
                
                else: # not the target cell we care about, set selfCellContribution equal to zero
                    for i,j,k in cell.PxByPyByPz:
                        gp_t = cell.gridpoints[i,j,k]
                        gp_t.selfCellContribution = 0.0 
        print('Done.')
                        
                        

    def extractConvolutionIntegrand_selfCell(self, containing=None): 
        '''
        Extract the leaves as a Nx5 array [ [x1,y1,z1,f1,w1], [x2,y2,z2,f2,w2], ... ] where f is the function being convolved
        '''
        
        leaves = []
        cellID = 0  
        for _,cell in self.masterList:
            if cell.leaf == True:
                if (   (containing==None)  or   
                    (  
                           ( (cell.xmin<containing[0]) and (cell.xmax>containing[0]) )  and 
                           ( (cell.ymin<containing[1]) and (cell.ymax>containing[1]) )  and  
                           ( (cell.zmin<containing[2]) and (cell.zmax>containing[2]) ) )
                    ):
                    for i,j,k in cell.PxByPyByPz:
                        gridpt = cell.gridpoints[i,j,k]
                        leaves.append( [gridpt.x, gridpt.y, gridpt.z, gridpt.f, cell.w[i,j,k], gridpt.selfCellContribution, cellID ] )
                cellID += 1
                            
        return np.array(leaves)
    
    def extractDenstiySecondaryMesh(self):
        '''
        Extract the leaves as a Nx5 array [ [x1,y1,z1,rho1,w1], [x2,y2,z2,rho2,w2], ... ]
        '''
#         print('Extracting the gridpoints from all leaves...')
        leaves = []
                
        for _,cell in self.masterList:
            if cell.leaf == True:
                for i,j,k in cell.PxByPyByPz_density:
                    densityPoint = cell.densityPoints[i,j,k]
                    leaves.append( [densityPoint.x, densityPoint.y, densityPoint.z, densityPoint.rho, cell.w_density[i,j,k], cell.volume ] )
                            
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
            
    def importPhiNewOnLeaves(self,phiNew):
        '''
        Import phi difference values, apply to leaves
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
                    gridpt.phiNew = phiNew[importIndex]
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
        Import V_coulomn values, apply to leaves
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
                    print('Warning: using gridpoints[1,1,1] as midpoint')
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
        
        phi0 = []
        phi1 = []
        phi2 = []
        phi3 = []
        phi4 = []
        phi5 = []
        phi6 = []
        phi7 = []
        phi8 = []
        phi9 = []
        for _,cell in self.masterList:
            if cell.leaf==True:
                for i,j,k in cell.PxByPyByPz:
                    gp = cell.gridpoints[i,j,k]
                    x.append(gp.x)
                    y.append(gp.y)
                    z.append(gp.z)
                    v.append(gp.v_eff)
                    phi0.append(gp.phi[0])
                    phi1.append(gp.phi[1])
                    phi2.append(gp.phi[2])
                    phi3.append(gp.phi[3])
                    phi4.append(gp.phi[4])
                    phi5.append(gp.phi[5])
                    phi6.append(gp.phi[6])
                    phi7.append(gp.phi[7])
                    phi8.append(gp.phi[8])
                    phi9.append(gp.phi[9])
        
        pointsToVTK(filename, np.array(x), np.array(y), np.array(z), data = 
                    {"V" : np.array(v), "Phi0" : np.array(phi0), "Phi1" : np.array(phi1),
                     "Phi2" : np.array(phi2), "Phi3" : np.array(phi3), "Phi4" : np.array(phi4),
                     "Phi5" : np.array(phi5), "Phi6" : np.array(phi6),
                     "Phi7" : np.array(phi7), "Phi8" : np.array(phi8), "Phi9" : np.array(phi9)})
        
        
#         phi10 = []
#         phi20 = []
#         phi21x = []
#         phi21y = []
#         phi21z = []
#         for _,cell in self.masterList:
#             if cell.leaf==True:
#                 for i,j,k in cell.PxByPyByPz:
#                     gp = cell.gridpoints[i,j,k]
#                     x.append(gp.x)
#                     y.append(gp.y)
#                     z.append(gp.z)
#                     v.append(gp.v_eff)
#                     phi10.append(gp.phi[0])
#                     phi20.append(gp.phi[1])
#                     phi21x.append(gp.phi[2])
#                     phi21y.append(gp.phi[3])
#                     phi21z.append(gp.phi[4])
#         
#         pointsToVTK(filename, np.array(x), np.array(y), np.array(z), data = 
#                     {"V" : np.array(v), "Phi10" : np.array(phi10), "Phi20" : np.array(phi20),
#                      "Phi21x" : np.array(phi21x), "Phi21y" : np.array(phi21y), "Phi21z" : np.array(phi21z)})
                
        
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
    

    
    
       
    