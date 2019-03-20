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
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.special import sph_harm
from scipy.optimize import broyden1, anderson, brentq
import pylibxc
import itertools
import os
import sys
import csv
# from numba.targets.options import TargetOptions
try:
    import vtk
except ModuleNotFoundError:
    pass
import time
import copy
try:
    from pyevtk.hl import pointsToVTK
except ImportError:
    sys.path.append('/home/njvaughn')
    try:
        from pyevtk.hl import pointsToVTK
    except ImportError:
        print("Wasn't able to import pyevtk.hl.pointsToVTK, even after appending '/home/njvaughn' to path.")
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
    
    
    def __init__(self, xmin,xmax,px,ymin,ymax,py,zmin,zmax,pz,nElectrons,nOrbitals,additionalDepthAtAtoms,minDepth,gaugeShift=0.0,
                 coordinateFile='',smoothingEps=0.0,inputFile='',exchangeFunctional="LDA_X",correlationFunctional="LDA_C_PZ",
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
#         self.PxByPyByPz = [element for element in itertools.product(range(self.px),range(self.py),range(self.pz))]
        self.nElectrons = nElectrons
        self.nOrbitals = nOrbitals
        self.gaugeShift = gaugeShift
        self.additionalDepthAtAtoms = additionalDepthAtAtoms
        self.minDepth = minDepth
        self.maxDepthAchieved = 0
        
        self.coordinateFile = coordinateFile
        
        
#         self.mixingParameter=0.5  # (1-mixingParam)*rhoNew
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
        if np.shape(atomData)==(5,):
            self.atoms = np.empty((1,),dtype=object)
            atom = Atom(atomData[0],atomData[1],atomData[2],atomData[3],atomData[4],smoothingEpsilon=smoothingEps)
            self.atoms[0] = atom
        else:
            self.atoms = np.empty((len(atomData),),dtype=object)
            for i in range(len(atomData)):
                atom = Atom(atomData[i,0],atomData[i,1],atomData[i,2],atomData[i,3],atomData[i,4],smoothingEpsilon=smoothingEps)
                self.atoms[i] = atom
        
        # generate gridpoint objects.  
        xvec = ChebyshevPoints(self.xmin,self.xmax,self.px)
        yvec = ChebyshevPoints(self.ymin,self.ymax,self.py)
        zvec = ChebyshevPoints(self.zmin,self.zmax,self.pz)
        gridpoints = np.empty((px,py,pz),dtype=object)


        for i in range(self.px):
            for j in range(self.py):
                for k in range(self.pz):
#         for i, j, k in self.PxByPyByPz:
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
#         self.initialDivideBasedOnNuclei(coordinateFile)
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
        
        self.T = 200
#         KB = 8.6173303e-5/27.211386
        KB = 1/315774.6
        self.sigma = self.T*KB
        
        
        
        eF = brentq(self.fermiObjectiveFunction, self.orbitalEnergies[0], 1, xtol=1e-14)
#         eF = brentq(self.fermiObjectiveFunction, self.orbitalEnergies[0], 1)
        print('Fermi energy: ', eF)
        exponentialArg = (self.orbitalEnergies-eF)/self.sigma
        self.occupations = 2*1/(1+np.exp( exponentialArg ) )  # these are # of electrons, not fractional occupancy.  Hence the 2*
        print('Occupations: ', self.occupations)
    
    
    def computeOrbitalMoments(self):
        for m in range(self.nOrbitals):
            x1 = 0
            y1 = 0
            z1 = 0
            x2 = 0
            y2 = 0
            z2 = 0
            
            for _,cell in self.masterList:
                if cell.leaf == True:
                    for i,j,k in cell.PxByPyByPz:
                        gp = cell.gridpoints[i,j,k]
                        
                        x1 += gp.phi[m]*gp.x*cell.w[i,j,k]
                        y1 += gp.phi[m]*gp.y*cell.w[i,j,k]
                        z1 += gp.phi[m]*gp.z*cell.w[i,j,k]
                        x2 += gp.phi[m]*gp.x**2*cell.w[i,j,k]
                        y2 += gp.phi[m]*gp.y**2*cell.w[i,j,k]
                        z2 += gp.phi[m]*gp.z**2*cell.w[i,j,k]
            
            print('\nOrbital ', m, ' moments:')
            print('x1 = ', x1)
            print('y1 = ', y1)
            print('z1 = ', z1)
            print('x2 = ', x2)
            print('y2 = ', y2)
            print('z2 = ', z2, '\n')
            
        return
        
        

            
    def initialDivideBasedOnNuclei(self, coordinateFile,maxLevels=99):
        
        def refineToMinDepth(self,Cell):
            # Divide the root to the minimum depth, BEFORE dividing at the nuclear positions
            
            if (Cell.level) < self.minDepth:
                print('Dividing cell %s because it is at depth %i.' %(Cell.uniqueID, Cell.level))
                xdiv = (Cell.xmax + Cell.xmin)/2   
                ydiv = (Cell.ymax + Cell.ymin)/2   
                zdiv = (Cell.zmax + Cell.zmin)/2   
                Cell.divide(xdiv, ydiv, zdiv)
                
                (ii,jj,kk) = np.shape(Cell.children)
                for i in range(ii):
                    for j in range(jj):
                        for k in range(kk):
                            refineToMinDepth(self, Cell.children[i,j,k])
 
        
        def recursiveDivideByAtom(self,Atom,Cell):
            # Atom is in this cell.  Check if this cell has children.  If so, find the child that contains
            # the atom.  If not, divideInto8 the cell.
            if hasattr(Cell, "children"):
                
                (ii,jj,kk) = np.shape(Cell.children)
                for i in range(ii):
                    for j in range(jj):
                        for k in range(kk):
#                 for i,j,k in TwoByTwoByTwo: # this should catch cases where atom is on the boundary of a previous cut
                            if ( (Atom.x <= Cell.children[i,j,k].xmax) and (Atom.x >= Cell.children[i,j,k].xmin) ):
                                if ( (Atom.y <= Cell.children[i,j,k].ymax) and (Atom.y >= Cell.children[i,j,k].ymin) ):
                                    if ( (Atom.z <= Cell.children[i,j,k].zmax) and (Atom.z >= Cell.children[i,j,k].zmin) ): 
#                                         print('Atom located inside or on the boundary of child ', Cell.children[i,j,k].uniqueID, '. Calling recursive divide.')
                                        recursiveDivideByAtom(self, Atom, Cell.children[i,j,k])

  
            else:  # sets the divideInto8 location.  If atom is on the boundary, sets divideInto8 location to None for that dimension
#                 print('Atom is contained in cell ', Cell.uniqueID,' which has no children.  Divide at atomic location.')
                if ( (Atom.x < Cell.xmax) and (Atom.x > Cell.xmin) and 
                     (Atom.y < Cell.ymax) and (Atom.y > Cell.ymin) and
                     (Atom.z < Cell.zmax) and (Atom.z > Cell.zmin) ):
                    print('Dividing cell %s because atom is in interior.' %(Cell.uniqueID))
                    xdiv = Atom.x
                    ydiv = Atom.y
                    zdiv = Atom.z
#                     if ( (Atom.x == Cell.xmax) or (Atom.x == Cell.xmin) ):
#                         xdiv = None
#                     if ( (Atom.y == Cell.ymax) or (Atom.y == Cell.ymin) ):
#                         ydiv = None
#                     if ( (Atom.z == Cell.zmax) or (Atom.z == Cell.zmin) ):
#                         zdiv = None
        
        
#                     if ( (xdiv!=None) or (ydiv!=None) or (zdiv!=None) ):    
                    Cell.divide(xdiv, ydiv, zdiv)
            
                    leafCount = 0
                    for _,cell in self.masterList:
                        if cell.leaf==True:
                            leafCount += 1
                    print('There are now %i leaf cells.' %leafCount)
                    (ii,jj,kk) = np.shape(Cell.children)
                    for i in range(ii):
                        for j in range(jj):
                            for k in range(kk):
                                recursiveDivideByAtom(self, Atom, Cell.children[i,j,k])
        
                else:   
                    if ( ( (Atom.x == Cell.xmax) or (Atom.x == Cell.xmin) )  and 
                         ( (Atom.y == Cell.ymax) or (Atom.y == Cell.ymin) ) and
                         ( (Atom.z == Cell.zmax) or (Atom.z == Cell.zmin) ) ): # atom is at a vertex.
                        if Cell.level < self.maxDepthAtAtoms: # But have we gone deep enough?
                            print('Dividing cell %s because atom is at corner and it is at depth %i' %(Cell.uniqueID,Cell.level))
                            xdiv = Cell.xmid
                            ydiv = Cell.ymid
                            zdiv = Cell.zmid
                            Cell.divide(xdiv, ydiv, zdiv)
                            (ii,jj,kk) = np.shape(Cell.children)
                            for i in range(ii):
                                for j in range(jj):
                                    for k in range(kk):
                                        recursiveDivideByAtom(self, Atom, Cell.children[i,j,k])
        
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
        
        
#         refineToMinDepth(self,self.root)
#         
#         cellCount = 0
#         for _,cell in self.masterList:
#             if cell.leaf==True:
#                 cellCount += 1
#         print('Number of cell after minDepth refine: ', cellCount)
        
        self.maxDepthAtAtoms = self.additionalDepthAtAtoms + self.maxDepthAchieved
        print('Refining an additional %i levels, from %i to %i' %(self.additionalDepthAtAtoms,self.maxDepthAchieved, self.maxDepthAtAtoms ))
        self.nAtoms = 0
        for atom in self.atoms:
            recursiveDivideByAtom(self,atom,self.root)
            self.nAtoms += 1
            
        self.computeNuclearNuclearEnergy()
#         
# #         # Reset all cells to level 1.  These divides shouldnt count towards its depth.    Do this BEFORE or AFTER aspect ratio divide?  Unclear.
# #         for _,cell in self.masterList:
# #             if cell.leaf==True:
# #                 cell.level = 1
# 
#         # Reset all cells to level minDepth.  These divides shouldnt count towards its depth.    Do this BEFORE or AFTER aspect ratio divide?  Unclear.
#         for _,cell in self.masterList:
#             if cell.leaf==True:
#                 cell.level = self.minDepth
#         
# #         self.exportMeshVTK('/Users/nathanvaughn/Desktop/aspectRatioBefore2.vtk')
#         for _,cell in self.masterList:
#             if cell.leaf==True:
#                 cell.divideIfAspectRatioExceeds(4) #283904 for aspect ratio 1.5, but 289280 for aspect ratio 10.0.  BUT, for 9.5, 8, 4, and so on, there are less quad points than 2.0.  So maybe not a bug 
#         
#         leafCount = 0
#         for _,cell in self.masterList:
#             if cell.leaf==True:
#                 leafCount += 1
#         print('After aspect ratio divide there are %i leaf cells.' %leafCount)
#         
#         # Now reset level to self.minDepth for any cell near an atom (which might not be 1 anymore because of aspect ratio divisions)
#         for _,cell in self.masterList:
#             if cell.leaf==True:
#                 for atom in self.atoms:
#                     rsq = (cell.xmid-atom.x)**2 + (cell.ymid-atom.y)**2 + (cell.zmid-atom.z)**2
#                     if rsq < 4:
#                         cell.level = self.minDepth
# #         # Reset all cells to level 1.  These divides shouldnt count towards its depth.  
# #         for _,cell in self.masterList:
# #             if cell.leaf==True:
# #                 cell.level = 1
#         
# #         print('Dividing adjacent to nuclei')  
# #         for atom in self.atoms:
# #             refineToMaxDepth(self,atom,self.root)
                
      
    def initializeOrbitalsRandomly(self,targetOrbital=None):
        timer = Timer()
        timer.start()
        if targetOrbital==None:
            print('Initializing all orbitals randomly...')
            for _,cell in self.masterList:
                if cell.leaf==True:
                    for i,j,k in cell.PxByPyByPz:
                        for m in range(self.nOrbitals):
                            gp = cell.gridpoints[i,j,k]
    #                         gp.phi[m] = np.sin(gp.x)/(abs(gp.x)+abs(gp.y)+abs(gp.z))/(m+1)
                            gp.phi[m] = np.random.rand(1)
                            
        else:
            print('Initializing orbital ',targetOrbital,' randomly...')
            for _,cell in self.masterList:
                if cell.leaf==True:
                    for i,j,k in cell.PxByPyByPz:
#                         for m in range(self.nOrbitals):
                        gp = cell.gridpoints[i,j,k]
#                         gp.phi[m] = np.sin(gp.x)/(abs(gp.x)+abs(gp.y)+abs(gp.z))/(m+1)
                        gp.phi[targetOrbital] = np.random.rand(1)
                        
        timer.stop()
        print('Initializing orbitals randomly inside Tree Structure took %.3f seconds.' %timer.elapsedTime)
                        
                            
        
    
    def initializeDensityFromAtomicDataExternally(self):
        timer = Timer()
        timer.start()
        
        sources = self.extractLeavesDensity()
        x = sources[:,0]
        y = sources[:,1]
        z = sources[:,2]
        rho = np.zeros(len(x))
        
        totalElectrons = 0
        for atom in self.atoms:
            totalElectrons += atom.atomicNumber
            r = np.sqrt( (x-atom.x)**2 + (y-atom.y)**2 + (z-atom.z)**2 )
#             for i in range(len(r)):
#                 try:
#                     rho[i] += atom.interpolators['density'](r[i])
#                 except ValueError:
#                     rho[i] += 0.0   # if outside the interpolation range, assume 0.
            try:
                rho += atom.interpolators['density'](r)
            except ValueError:
                rho += 0.0   # if outside the interpolation range, assume 0.
                
            print("max density: ", max(abs(rho)))
            self.importDensityOnLeaves(rho)
#             print('Should I normalize density??')
#             self.normalizeDensityToValue(totalElectrons) 
            sources = self.extractLeavesDensity()
            rho = np.copy(sources[:,3])
        
#         print("max density: ", max(abs(rho)))
#         self.importDensityOnLeaves(rho) 

        self.normalizeDensityToValue(totalElectrons)
         
        timer.stop()
        print('Initializing density EXTERNALLY took %.3f seconds.' %timer.elapsedTime)  
        
       
    
    def initializeDensityFromAtomicData(self):
        timer = Timer()
        timer.start()
        
        for _,cell in self.masterList:
            if cell.leaf==True:
                for i,j,k in cell.PxByPyByPz:
                    gp = cell.gridpoints[i,j,k]
                    gp.rho = 0.0
                    for atom in self.atoms:
                        r = np.sqrt( (gp.x-atom.x)**2 + (gp.y-atom.y)**2 + (gp.z-atom.z)**2 )
                        try:
                            gp.rho += atom.interpolators['density'](r)
                        except ValueError:
                            gp.rho += 0.0   # if outside the interpolation range, assume 0.
                            
        timer.stop()
        print('Initializing density inside Tree Structure took %.3f seconds.' %timer.elapsedTime)
                
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
                            
    def initializeOrbitalsFromAtomicDataExternally(self,onlyFillOne=False): 
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
        
#         print('Setting second atom nOrbitals to 2 for carbon monoxide.  Also setting tree.nOrbitals to 7')
#         self.atoms[1].nAtomicOrbitals = 2
#         self.nOrbitals = 7
        

    
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
                        
                        sources = self.extractPhi(orbitalIndex)
                        dx = sources[:,0]-atom.x
                        dy = sources[:,1]-atom.y
                        dz = sources[:,2]-atom.z
                        phi = np.zeros(len(dx))
                        r = np.sqrt( dx**2 + dy**2 + dz**2 )
                        inclination = np.arccos(dz/r)
                        azimuthal = np.arctan2(dy,dx)
                        
                        if m<0:
                            Y = (sph_harm(m,ell,azimuthal,inclination) + (-1)**m * sph_harm(-m,ell,azimuthal,inclination))/np.sqrt(2) 
                        if m>0:
                            Y = 1j*(sph_harm(m,ell,azimuthal,inclination) - (-1)**m * sph_harm(-m,ell,azimuthal,inclination))/np.sqrt(2)
#                                     if ( (m==0) and (ell>1) ):
                        if ( m==0 ):
                            Y = sph_harm(m,ell,azimuthal,inclination)
#                                     if ( (m==0) and (ell<=1) ):
#                                         Y = 1
                        if np.max( abs(np.imag(Y)) ) > 1e-14:
                            print('imag(Y) ', np.imag(Y))
                            return
#                                     Y = np.real(sph_harm(m,ell,azimuthal,inclination))
                        phi = atom.interpolators[psiID](r)*np.real(Y)
                        
                        
                        self.importPhiOnLeaves(phi, orbitalIndex)
                        self.normalizeOrbital(orbitalIndex)
                        
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
            print('Filling extra orbitals with random initial data.')
            for ii in range(orbitalIndex, self.nOrbitals):
                self.initializeOrbitalsRandomly(targetOrbital=ii)
#                 self.orthonormalizeOrbitals(targetOrbital=ii)
        if orbitalIndex > self.nOrbitals:
            print("Filled too many orbitals, somehow.  That should have thrown an error and never reached this point.")
                        
        timer.stop()
        print('Initializing orbitals EXTERNAL to Tree Structure took %.3f seconds.' %timer.elapsedTime)
        
        for m in range(self.nOrbitals):
            self.normalizeOrbital(m)
            
            
            
                           
    
    def initializeOrbitalsFromAtomicData_deprecated(self,onlyFillOne=False):
        
        
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
#         print('Setting second atom nOrbitals to 2 for carbon monoxide.  Also setting tree.nOrbitals to 7')
#         self.atoms[1].nAtomicOrbitals = 2
#         self.nOrbitals = 7
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
                                for i,j,k in cell.PxByPyByPz:
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
            print('Filling extra orbitals with random initial data.')
            for ii in range(orbitalIndex, self.nOrbitals):
                self.initializeOrbitalsRandomly(targetOrbital=ii)
                self.orthonormalizeOrbitals(targetOrbital=ii)
        if orbitalIndex > self.nOrbitals:
            print("Filled too many orbitals, somehow.  That should have thrown an error and never reached this point.")
                        
        timer.stop()
        print('Initializing orbitals inside Tree Structure took %f.3 seconds.' %timer.elapsedTime)
        
        for m in range(self.nOrbitals):
            self.normalizeOrbital(m)
    

        

 
        
    def buildTree(self,maxLevels, divideCriterion, divideParameter1, divideParameter2=0.0, divideParameter3=0.0, divideParameter4=0.0, initializationType='atomic',printNumberOfCells=False, printTreeProperties = True, onlyFillOne=False): # call the recursive divison on the root of the tree
        # max depth returns the maximum depth of the tree.  maxLevels is the limit on how large the tree is allowed to be,
        # regardless of division criteria
        # N is roughly the number of grid points.  It is used to generate the density function.
        
        divideParameter = divideParameter1 # for methods that use only one divide parameter.
        timer = Timer()
        def recursiveDivide(self, Cell, maxLevels, divideCriterion, divideParameter, levelCounter, maxDepthCounter, printNumberOfCells, maxDepthAchieved=0, minDepthAchieved=100):
            levelCounter += 1
            
            if hasattr(Cell, "children"):
#                 print('Cell already has children')
                (ii,jj,kk) = np.shape(Cell.children)

                for i in range(ii):
                    for j in range(jj):
                        for k in range(kk):
                            maxDepthAchieved, minDepthAchieved, levelCounter, maxDepthCounter = recursiveDivide(self,Cell.children[i,j,k], 
                                                                                maxLevels, divideCriterion, divideParameter, 
                                                                                levelCounter, maxDepthCounter, printNumberOfCells, maxDepthAchieved, 
                                                                                minDepthAchieved)
            
            elif Cell.level < maxLevels:
#             elif currentLevel < maxLevels:
                
#                 if currentLevel < self.minDepth:
                if Cell.level < self.minDepth:
                    Cell.divideFlag = True 
#                     print('dividing cell ', Cell.uniqueID, ' because it is below the minimum level')
                else:  
                    if ( (divideCriterion == 'LW1') or (divideCriterion == 'LW2') or (divideCriterion == 'LW3') or (divideCriterion == 'LW3_modified') or 
                         (divideCriterion == 'LW4') or (divideCriterion == 'LW5') or(divideCriterion == 'Phani') 
                         or (divideCriterion == 'Krasny_density') or (divideCriterion == 'Nathan_density')  ):
#                         print('checking divide criterion for cell ', Cell.uniqueID)
                        Cell.checkIfAboveMeshDensity(divideParameter,divideCriterion)  
                    elif divideCriterion=='Biros':
                        Cell.checkIfChebyshevCoefficientsAboveTolerance(divideParameter)
                    elif divideCriterion=='BirosCombined':
                        Cell.checkIfChebyshevCoefficientsAboveTolerance_DensityAndWavefunctions(divideParameter)
                    elif divideCriterion=='BirosK':
                        Cell.checkIfChebyshevCoefficientsAboveTolerance_allIndicesAboveQ(divideParameter)
                    elif divideCriterion=='BirosN':
                        Cell.checkIfChebyshevCoefficientsAboveTolerance_anyIndicesAboveQ_rho_sqrtRho(divideParameter1, divideParameter2)
                    elif divideCriterion=='BirosG':
#                         Cell.checkIfChebyshevCoefficientsAboveTolerance_anyIndicesAboveQ_sumOfWavefunctions(divideParameter)
                        Cell.checkIfChebyshevCoefficientsAboveTolerance_anyIndicesAboveQ_sumOfWavefunctions(divideParameter)
                    elif divideCriterion=='BirosGN':
                        Cell.checkIfChebyshevCoefficientsAboveTolerance_anyIndicesAboveQ_psi_or_rho(divideParameter1, divideParameter2)
                    elif divideCriterion=='BirosGN2':
                        Cell.checkIfChebyshevCoefficientsAboveTolerance_anyIndicesAboveQ_psi_or_rho_or_v(divideParameter)
                    elif divideCriterion=='Krasny':
                        Cell.checkWavefunctionVariation(divideParameter1, divideParameter2, divideParameter3, divideParameter4)
                    elif divideCriterion=='Krasny_Vext':
                        Cell.checkWavefunctionVariation_Vext(divideParameter1, divideParameter2, divideParameter3, divideParameter4)
                    elif divideCriterion=='Krasny_log_rho':
                        Cell.checkLogDensityVariation(divideParameter1, divideParameter2, divideParameter3, divideParameter4)
                    elif divideCriterion=='Nathan':
                        Cell.checkDensityIntegral(divideParameter1, divideParameter2)
                    elif divideCriterion=='Nathan2':
                        Cell.checkMeshDensity_Nathan(divideParameter1, divideParameter2)
                    elif divideCriterion=='NathanNearFar':
                        Cell.splitNearAndFar(divideParameter1, divideParameter2, divideParameter3, divideParameter4)
                    elif divideCriterion == 'basic_density':
                        Cell.compareMeshDensityToDensity(divideParameter)
                    elif divideCriterion=='Krasny_interpolate':
                        Cell.checkDensityInterpolation(divideParameter1, divideParameter2, divideParameter3, divideParameter4)
                        
                    elif divideCriterion=='ParentChildrenIntegral':
                        Cell.refineByCheckingParentChildrenIntegrals(divideParameter1, divideParameter2, divideParameter3)  
                    
                    else:                        
                        Cell.checkIfCellShouldDivide(divideParameter)
                    
                if Cell.divideFlag == True:
                    xdiv = (Cell.xmax + Cell.xmin)/2   
                    ydiv = (Cell.ymax + Cell.ymin)/2   
                    zdiv = (Cell.zmax + Cell.zmin)/2   
                    Cell.divide(xdiv, ydiv, zdiv, printNumberOfCells)

#                     for i,j,k in TwoByTwoByTwo:  # what if there aren't 8 children?
                    (ii,jj,kk) = np.shape(Cell.children)
                    for i in range(ii):
                        for j in range(jj):
                            for k in range(kk):
                                maxDepthAchieved, minDepthAchieved, levelCounter, maxDepthCounter = recursiveDivide(self,Cell.children[i,j,k], maxLevels, divideCriterion, divideParameter, levelCounter, maxDepthCounter, printNumberOfCells, maxDepthAchieved, minDepthAchieved)
                else:
                    minDepthAchieved = min(minDepthAchieved, Cell.level)
                    
            else: 
#                 Cell.initializeCellWavefunctions()
#                 print("Cell %s at max depth.  Cell volume**(1/3) = %f" %(Cell.uniqueID, Cell.volume**(1/3)))
                maxDepthCounter +=1
                       
            maxDepthAchieved = max(maxDepthAchieved, Cell.level)                                                                                                                                                       
            return maxDepthAchieved, minDepthAchieved, levelCounter, maxDepthCounter
        
        timer.start()
#         self.initialDivideBasedOnNuclei(self.coordinateFile)
        levelCounter=0
        maxDepthCounter=0
        self.maxDepthAchieved, self.minDepthAchieved, self.treeSize, self.maxDepthCounter = recursiveDivide(self, self.root, maxLevels, divideCriterion, divideParameter, levelCounter, maxDepthCounter, printNumberOfCells, maxDepthAchieved=0, minDepthAchieved=maxLevels )
        
        print('Number of cells at max depth: ', self.maxDepthCounter)
#         self.initialDivideBasedOnNuclei(self)
        self.initialDivideBasedOnNuclei(self.coordinateFile)
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
                for i,j,k in cell.PxByPyByPz:
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
        
        self.countCellsAtEachDepth()
        
        
                        
        for _,cell in self.masterList:
            for i,j,k in cell.PxByPyByPz:
                if hasattr(cell.gridpoints[i,j,k], "counted"):
                    cell.gridpoints[i,j,k].counted = None
         
        
        print('Number of gridpoints: ', self.numberOfGridpoints)

        self.computeDerivativeMatrices()
#         self.initializeDensityFromAtomicData()

        self.initializeDensityFromAtomicDataExternally()  # do this extrnal to the tree.  Roughly 10x faster than in the tree.
        
        ### INITIALIZE ORBTIALS AND DENSITY ####
                # Only need to do this if wavefunctions aren't set during adaptive refinement

        if initializationType=='atomic':
            if onlyFillOne == True:
                self.initializeOrbitalsFromAtomicDataExternally(onlyFillOne=True)
            else:
                self.initializeOrbitalsFromAtomicDataExternally()
        elif initializationType=='random':
            self.initializeOrbitalsRandomly()

        
        
        timer.stop()
                    
        if printTreeProperties == True: 
            print("Tree build completed. \n"
                  "Domain Size:                                 [%.1f, %.1f] \n"
                  "Divide Criterion:                            %s \n"
                  "Divide Parameter1:                           %1.2e \n"
                  "Divide Parameter2:                           %1.2e \n"
                  "Divide Parameter3:                           %1.2e \n"
                  "Divide Parameter4:                           %1.2e \n"
                  "Total Number of Cells:                       %i \n"
                  "Total Number of Leaf Cells:                  %i \n"
                  "Total Number of Gridpoints:                  %i \n"
                  "Minimum Depth                                %i levels \n"
                  "Maximum Depth from refinement scheme:        %i levels \n"
                  "Additional refinement to depth:              %i levels \n"
                  "Cell Order:                                  %i \n"
                  "Construction time:                           %.3g seconds."
                   
                  %(self.xmin, self.xmax, divideCriterion,divideParameter1,divideParameter2,divideParameter3,divideParameter4, self.treeSize, self.numberOfCells, self.numberOfGridpoints, self.minDepthAchieved,self.maxDepthAchieved, 
                    self.maxDepthAtAtoms, self.px, timer.elapsedTime))
            print('Closest gridpoint to origin: ', closestCoords)
            print('For a distance of: ', closestToOrigin)
            print('Part of a cell centered at: ', closestMidpoint) 

    
    def countCellsAtEachDepth(self):
        levelCounts = {}
        criteria1 = {}
        criteria2 = {}
        criteria3 = {}
        criteria4 = {}
        
        for level in range(self.minDepth, self.maxDepthAchieved):
            levelCounts[level] = 0
            criteria1[level] = 0
            criteria2[level] = 0
            criteria3[level] = 0
            criteria4[level] = 0
        for _,cell in self.masterList:
            if cell.leaf==True:
                if cell.level in levelCounts:
                    levelCounts[cell.level] += 1
                else:
                    levelCounts[cell.level] = 1
                    print('Added level %i to dictionary.' %cell.level)
                if hasattr(cell, "refineCause"):
                    if cell.refineCause==1:
                        if cell.level in criteria1:
                            criteria1[cell.level] += 1
                        else:
                            criteria1[cell.level] = 1
                            print('Added level %i to criteria1 dictionary.' %cell.level)
                    
                    if cell.refineCause==2:
                        if cell.level in criteria2:
                            criteria2[cell.level] += 1
                        else:
                            criteria2[cell.level] = 1
                            print('Added level %i to criteria2 dictionary.' %cell.level)
                    if cell.refineCause==3:
                        if cell.level in criteria3:
                            criteria3[cell.level] += 1
                        else:
                            criteria3[cell.level] = 1
                            print('Added level %i to criteria3 dictionary.' %cell.level)
                    if cell.refineCause==4:
                        if cell.level in criteria4:
                            criteria4[cell.level] += 1
                        else:
                            criteria4[cell.level] = 1
                            print('Added level %i to criteria4 dictionary.' %cell.level)
                        
        
        print('Number of cells at each level: ')
        print(levelCounts)
        
        self.levelCounts=levelCounts
        self.criteria1=criteria1
        self.criteria2=criteria2
        self.criteria3=criteria3
        self.criteria4=criteria4
        return
        
    
    """
    UPDATE DENSITY AND EFFECTIVE POTENTIAL AT GRIDPOINTS
    """
    def updateVxcAndVeffAtQuadpoints(self):
#         print('Warning: v_xc is zeroed out')
        
        def CellupdateVxcAndVeff(cell,exchangeFunctional, correlationFunctional):
            '''
            After density is updated the convolution gets called to update V_hartree.
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

    def updateDensityAtQuadpoints(self, mixingScheme='None'):
        def CellUpdateDensity(cell,mixingScheme):
            for i,j,k in cell.PxByPyByPz:
                newRho = 0
                for m in range(self.nOrbitals):
                    newRho += cell.tree.occupations[m] * cell.gridpoints[i,j,k].phi[m]**2
                if mixingScheme=='None':
                    cell.gridpoints[i,j,k].rho = newRho
                    
                    ## Mixing has been taken care of externally.  This function is now only supposed to update density from wavefunctions, not handle mixing as well
                    
                    
#                 elif mixingScheme=='Simple':
#                     cell.gridpoints[i,j,k].rho = ( self.mixingParameter*cell.gridpoints[i,j,k].rho + 
#                         (1-self.mixingParameter)*newRho )
                else: 
                    print('Not a valid density mixing scheme.')
                    return
            
            
        for _,cell in self.masterList:
            if cell.leaf==True:
                CellUpdateDensity(cell,mixingScheme)
     
    
    def normalizeDensityToValue(self,value):
        A = 0.0
        for _,cell in self.masterList:
            if cell.leaf==True:
                for i,j,k in cell.PxByPyByPz:
                    gp = cell.gridpoints[i,j,k]
                    A += gp.rho * cell.w[i,j,k]
        print('Integrated density before normalization: ', A)
        for _,cell in self.masterList:
            if cell.leaf==True:
                for i,j,k in cell.PxByPyByPz:
                    gp = cell.gridpoints[i,j,k]
                    gp.rho /= A
                    gp.rho *= value
        A = 0.0
        for _,cell in self.masterList:
            if cell.leaf==True:
                for i,j,k in cell.PxByPyByPz:
                    gp = cell.gridpoints[i,j,k]
                    A += gp.rho * cell.w[i,j,k]
        print('Integrated density after normalization: ', A)
                
    
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
        V_hartree = 0.0
        E_x = 0.0
        E_c = 0.0
        E_electronNucleus = 0.0
        
        
        for _,cell in self.masterList:
            if cell.leaf == True:
                V_x += integrateCellDensityAgainst__(cell,'v_x')
                V_c += integrateCellDensityAgainst__(cell,'v_c')
                V_hartree += integrateCellDensityAgainst__(cell,'v_hartree')
                E_x += integrateCellDensityAgainst__(cell,'epsilon_x')
                E_c += integrateCellDensityAgainst__(cell,'epsilon_c')
                E_electronNucleus += integrateCellDensityAgainst__(cell,'v_ext')
                
        self.totalVx = V_x
        self.totalVc = V_c
        self.totalEhartree = 1/2*V_hartree
#         self.totalElectrostatic = -1/2*V_hartree - V_x - V_c + self.nuclearNuclear + self.totalOrbitalPotential
        self.totalElectrostatic = 1/2*V_hartree + self.nuclearNuclear + E_electronNucleus
        self.totalEx = E_x
        self.totalEc = E_c
        self.totalVext = E_electronNucleus
#         print('Total V_xc : ')
        
        print('Electrostatic Energies:')
        print('Hartree:         ', 1/2*V_hartree)
        print('External:        ', E_electronNucleus)
        print('Nuclear-Nuclear: ', self.nuclearNuclear)
#         print('Sanity check...')
#         print('Band minus kinetic: ', self.totalBandEnergy - self.totalKinetic)
#         print('Electrostatic minus external and Nuclear plus V_x and V_c: ', self.totalElectrostatic - self.nuclearNuclear - E_electronNucleus + self.totalVc + self.totalVx)
        
#         self.totalPotential = -1/2*V_hartree + E_xc - V_xc 
#         self.totalPotential = -1/2*V_hartree + E_x + E_c - V_x - V_c + self.nuclearNuclear
        self.totalPotential = self.totalElectrostatic +  E_x + E_c # - V_x - V_c 
                
        
       
    def computeBandEnergy(self):
        # sum over the kinetic energies of all orbitals
        print('Computing band energy.  Current orbital energies are: ', self.orbitalEnergies)
        self.totalBandEnergy = 0.0
        for i in range(self.nOrbitals):
            self.totalBandEnergy += self.occupations[i]*(self.orbitalEnergies[i] - self.gaugeShift) 
        
    
    
    def updateTotalEnergy(self,gradientFree):
        self.computeBandEnergy()
        self.computeTotalPotential()
        self.totalKinetic = self.totalBandEnergy - 2*self.totalEhartree - self.totalVx - self.totalVc - self.totalVext
        
        if gradientFree==True:
            self.E = self.totalBandEnergy - self.totalEhartree + self.totalEx + self.totalEc - self.totalVx - self.totalVc + self.nuclearNuclear
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
#                 val = np.random.rand(1)
                for i,j,k in cell.PxByPyByPz:
                    val = np.random.rand(1)
                    gp = cell.gridpoints[i,j,k]
#                     r = np.sqrt(gp.x*gp.x + gp.y*gp.y + gp.z*gp.z)
#                     gp.phi[m] = val/r
                    gp.phi[m] = val
                    
    def increaseNumberOfWavefunctionsByOne(self):
        
        for _,cell in self.masterList:
            for i,j,k in cell.PxByPyByPz:
                gp = cell.gridpoints[i,j,k]
                gp.phi = np.append(gp.phi,1.0)
        
#         self.scrambleOrbital(-1)
        self.orthonormalizeOrbitals(targetOrbital = -1)
        self.occupations = np.append(self.occupations, 0.0)
        self.orbitalEnergies = np.append(self.orbitalEnergies, self.gaugeShift-0.1)
        self.nOrbitals += 1
    
    def softenOrbital(self,m):
        print('Softening orbital ', m)
        # randomize orbital because its energy went > Vgauge
        for _,cell in self.masterList:
            if cell.leaf==True:
                for i,j,k in cell.PxByPyByPz:
                    gp = cell.gridpoints[i,j,k]
                    r = np.sqrt(gp.x*gp.x + gp.y*gp.y + gp.z*gp.z)
                    gp.phi[m] *= np.exp(-r)
                    
    def zeroOutOrbital(self,m):
#         print('Zeroing orbital ', m)
        print('setting orbital %i to exp(-r).'%m)
        # randomize orbital because its energy went > Vgauge
        for _,cell in self.masterList:
            if cell.leaf==True:
                for i,j,k in cell.PxByPyByPz:
                    gp = cell.gridpoints[i,j,k]
                    r = np.sqrt(gp.x*gp.x + gp.y*gp.y + gp.z*gp.z)
                    gp.phi[m] = np.exp(-r)
        self.normalizeOrbital(m)
        
    def resetOrbitalij(self,m,n):
        # set orbital m equal to orbital n
        for _,cell in self.masterList:
            if cell.leaf==True:
                for i,j,k in cell.PxByPyByPz:
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
    
    
    def updateOrbitalEnergies(self,newOccupations=False,correctPositiveEnergies=False,sortByEnergy=False,targetEnergy=None, saveAsReference=False, sortOrder=None):
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



    def sortOrbitalsAndEnergies(self, order=None):
        if order==None:
            newOrder = np.argsort(self.orbitalEnergies)
        elif order != None:
            newOrder = order
        oldEnergies = np.copy(self.orbitalEnergies)
        for m in range(self.nOrbitals):
            self.orbitalEnergies[m] = oldEnergies[newOrder[m]]
        print('Sorted eigenvalues: ', self.orbitalEnergies)
        print('New order: ', newOrder)
        for _,cell in self.masterList:
            if cell.leaf==True:
                for i,j,k in cell.PxByPyByPz:
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
        for _,cell in self.masterList:
            if cell.leaf == True:
                for i,j,k in self.PxByPyByPz:
                    gridpt = cell.gridpoints[i,j,k]
                    phiComputed[i,j,k] = gridpt.phi
                    phiAnalytic[i,j,k] = normalizationFactor*trueWavefunction(energyLevel,gridpt.x,gridpt.y,gridpt.z)
                errorsIfSameSign.append( np.sum( (phiAnalytic-phiComputed)**2*cell.w ))
                errorsIfDifferentSign.append( np.sum( (phiAnalytic+phiComputed)**2*cell.w ))
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
                for i,j,k in cell.PxByPyByPz:
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
                for i,j,k in cell.PxByPyByPz:
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
#                 for i,j,k in cell.PxByPyByPz:
#                     cell.gridpoints[i,j,k].orthogonalized = False
#         
#         """ Subtract the projection, flip the flag """
#         for _,cell in self.masterList:
#             if cell.leaf==True:
#                 for i,j,k in cell.PxByPyByPz:
#                     gridpoint = cell.gridpoints[i,j,k]
#                     if gridpoint.orthogonalized == False:
#                         gridpoint.phi -= B*gridpoint.finalWavefunction[n]
#                         gridpoint.orthogonalized = True
    
    def copyPhiNewToArray(self, targetOrbital):
        for _,cell in self.masterList:
            if cell.leaf == True:
                for i,j,k in cell.PxByPyByPz:
                    cell.gridpoints[i,j,k].phi[targetOrbital] = np.copy( cell.gridpoints[i,j,k].phiNew )
    
    
    def normalizePhiNew(self):
        """ Enforce integral phi*2 dxdydz == 1 """
        A = 0.0        
        for _,cell in self.masterList:
            if cell.leaf == True:
                for i,j,k in cell.PxByPyByPz:
                    A += cell.gridpoints[i,j,k].phiNew**2*cell.w[i,j,k]

        """ Rescale wavefunction values, flip the flag """
        for _,cell in self.masterList:
            if cell.leaf==True:
                for i,j,k in cell.PxByPyByPz:
                        cell.gridpoints[i,j,k].phiNew /= np.sqrt(A)
        
    
    def orthonormalizePhiNew(self, targetOrbital):
        
        def orthogonalizePhiNew(tree,targetOrbital,n):
            
#             print('Orthogonalizing phiNew %i against orbital %i' %(targetOrbital,n))
            """ Compute the overlap, integral phi_r * phi_s """
            B = 0.0
            for _,cell in tree.masterList:
                if cell.leaf == True:
                    for i,j,k in cell.PxByPyByPz:
                        phiNew = cell.gridpoints[i,j,k].phiNew
                        phi_n = cell.gridpoints[i,j,k].phi[n]
                        B += phiNew*phi_n*cell.w[i,j,k]

            """ Subtract the projection """
            for _,cell in tree.masterList:
                if cell.leaf==True:
                    for i,j,k in cell.PxByPyByPz:
                        gridpoint = cell.gridpoints[i,j,k]
                        gridpoint.phiNew -= B*gridpoint.phi[n]
                        
        def normalizePhiNew(tree,m):
        
            """ Enforce integral phi*2 dxdydz == 1 """
            A = 0.0        
            for _,cell in tree.masterList:
                if cell.leaf == True:
                    for i,j,k in cell.PxByPyByPz:
                        A += cell.gridpoints[i,j,k].phiNew**2*cell.w[i,j,k]
    
            """ Rescale wavefunction values, flip the flag """
            for _,cell in tree.masterList:
                if cell.leaf==True:
                    for i,j,k in cell.PxByPyByPz:
                            cell.gridpoints[i,j,k].phiNew /= np.sqrt(A)
                            
        print('Orthogonalizing phiNew, but not normalizing.')
#         print('Orthonormalizing phiNew.')
        for n in range(targetOrbital):
            orthogonalizePhiNew(self,targetOrbital,n)
#             normalizePhiNew(self,targetOrbital)
        
        
        
        
        
                        
    def orthonormalizeOrbitals(self, targetOrbital=None, external=False):
        
        
        def orthogonalizeOrbitals_external(tree,phi_m,phi_n,weights,targetOrbital):
            
#             sources = self.extractPhi(m)
#             phi_m = sources[:,3]
#             sources = self.extractPhi(n)
#             phi_n = sources[:,3]
#             weights = sources[:,5]
            
            phi_m -= ( np.dot(phi_m,phi_n*weights) / np.dot(phi_n,phi_n*weights) )*phi_n
            phi_m /= np.sqrt( np.dot(phi_m,phi_m*weights) )
            self.importPhiOnLeaves(phi_m, targetOrbital)
            
        
        def orthogonalizeOrbitals(tree,m,n):
            
#             return
#             print('Orthogonalizing %i against %i' %(m,n))
#             print('Orthogonalizing orbital %i against %i' %(m,n))
            """ Compute the overlap, integral phi_r * phi_s """
            B = 0.0
            for _,cell in tree.masterList:
                if cell.leaf == True:
                    for i,j,k in cell.PxByPyByPz:
                        phi_m = cell.gridpoints[i,j,k].phi[m]
                        phi_n = cell.gridpoints[i,j,k].phi[n]
#                         if abs(phi_m*phi_n*cell.w[i,j,k])>1e-8:
#                             B += phi_m*phi_n*cell.w[i,j,k]
                        B += phi_m*phi_n*cell.w[i,j,k]
#             print('Overlap before orthogonalization: ', B)

            """ Subtract the projection """
            for _,cell in tree.masterList:
                if cell.leaf==True:
                    for i,j,k in cell.PxByPyByPz:
                        gridpoint = cell.gridpoints[i,j,k]
#                         if abs(phi_m*phi_n)>1e-8:
                        gridpoint.phi[m] -= B*gridpoint.phi[n]
            
#             B = 0.0
#             for _,cell in tree.masterList:
#                 if cell.leaf == True:
#                     for i,j,k in cell.PxByPyByPz:
#                         phi_m = cell.gridpoints[i,j,k].phi[m]
#                         phi_n = cell.gridpoints[i,j,k].phi[n]
#                         B += phi_m*phi_n*cell.w[i,j,k]
#             print('Overlap after orthogonalization: ', B)
        
        def normalizeOrbital(tree,m):
        
            """ Enforce integral phi*2 dxdydz == 1 """
            A = 0.0        
            for _,cell in tree.masterList:
                if cell.leaf == True:
                    for i,j,k in cell.PxByPyByPz:
                        A += cell.gridpoints[i,j,k].phi[m]**2*cell.w[i,j,k]
    
            """ Rescale wavefunction values, flip the flag """
            for _,cell in tree.masterList:
                if cell.leaf==True:
                    for i,j,k in cell.PxByPyByPz:
                            cell.gridpoints[i,j,k].phi[m] /= np.sqrt(A)
                            
#             A = 0.0        
#             for _,cell in tree.masterList:
#                 if cell.leaf == True:
#                     for i,j,k in cell.PxByPyByPz:
#                         A += cell.gridpoints[i,j,k].phi[m]**2*cell.w[i,j,k]
#             print('Integral of phi%i**2 = ' %m, A )
        
        print('DO NOT USE TREE METHOD FOR ORTHOGONALIZATION')
        print('DO NOT USE TREE METHOD FOR ORTHOGONALIZATION')
        print('DO NOT USE TREE METHOD FOR ORTHOGONALIZATION')
        print('DO NOT USE TREE METHOD FOR ORTHOGONALIZATION')
        print('DO NOT USE TREE METHOD FOR ORTHOGONALIZATION')
        if targetOrbital==None:
#         print('Orthonormalizing orbitals within tree structure up to orbital %i.' %maxOrbital)
            for m in range(self.nOrbitals):
                if external==True:
                    normalizeOrbital(self,m)
                    sources = self.extractPhi(m)
                    phi_m = np.copy(sources[:,3])
                    weights = np.copy(sources[:,5])
                    for n in range(m):
                        normalizeOrbital(self,n)
                        sources = self.extractPhi(n)
                        phi_n = np.copy(sources[:,3])
                        orthogonalizeOrbitals_external(self,phi_m,phi_n,weights,targetOrbital=m)
                    
                    
                else:
                    normalizeOrbital(self,m)
                    for n in range(m):
                        normalizeOrbital(self,n)
                        orthogonalizeOrbitals(self,m,n)
                        normalizeOrbital(self,m)
        else:
            for n in range(targetOrbital):
                if external==True:
#                     if self.orbitalEnergies[targetOrbital]> self.orbitalEnergies[n]:
                    sources = self.extractPhi(targetOrbital)
                    phi_m = np.copy(sources[:,3])
                    weights = np.copy(sources[:,5])
                    sources = self.extractPhi(n)
                    phi_n = np.copy(sources[:,3])
                    orthogonalizeOrbitals_external(self,phi_m,phi_n,weights,targetOrbital=targetOrbital)
#                     else:
#                         print('Not orthogonalizing orbital %i against %i because energy is lower.' %(targetOrbital,n))
                else:
                    normalizeOrbital(self,n)
                    orthogonalizeOrbitals(self,targetOrbital,n)
                    normalizeOrbital(self,targetOrbital)
#                     orthogonalizeOrbitals(self,targetOrbital,n)
            normalizeOrbital(self,targetOrbital)  # orthonormalize once more at the end (important for psi0)
            
        
    
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
            for i,j,k in cell.PxByPyByPz:
                cell.gridpoints[i,j,k].extracted = False
                
        for _,cell in self.masterList:
            if cell.leaf == True:
                for i,j,k in cell.PxByPyByPz:
                    gridpt = cell.gridpoints[i,j,k]
                    if gridpt.extracted == False:
                        leaves.append( [gridpt.x, gridpt.y, gridpt.z, gridpt.phi, gridpt.v_eff, cell.w[i,j,k] ] )
                        gridpt.extracted = True
                    

        for _,cell in self.masterList:
            for i,j,k in cell.PxByPyByPz:
                cell.gridpoints[i,j,k].extracted = False
                
        return np.array(leaves)
    
    def extractPhi(self, orbitalNumber):
        '''
        Extract the leaves as a Nx4 array [ [x1,y1,z1,psi1], [x2,y2,z2,psi2], ... ]
        '''
#         print('Extracting the gridpoints from all leaves...')
        leaves = []
#         for _,cell in self.masterList:
#             for i,j,k in cell.PxByPyByPz:
#                 cell.gridpoints[i,j,k].extracted = False
                
        for _,cell in self.masterList:
            if cell.leaf == True:
                for i,j,k in cell.PxByPyByPz:
                    gridpt = cell.gridpoints[i,j,k]
#                     if gridpt.extracted == False:
                    leaves.append( [gridpt.x, gridpt.y, gridpt.z, gridpt.phi[orbitalNumber], gridpt.v_eff, cell.w[i,j,k], cell.volume ] )
#                         gridpt.extracted = True
                    

#         for _,cell in self.masterList:
#             for i,j,k in cell.PxByPyByPz:
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
#             for i,j,k in cell.PxByPyByPz:
#                 cell.gridpoints[i,j,k].phiImported = False
        importIndex = 0        
        for _,cell in self.masterList:
            if cell.leaf == True:
                for i,j,k in cell.PxByPyByPz:
                    gridpt = cell.gridpoints[i,j,k]
#                     if gridpt.phiImported == False:
                    gridpt.phi[orbitalNumber] = phiNew[importIndex]
#                         gridpt.phiImported = True
                    importIndex += 1
                    
#         for _,cell in self.masterList:
#             for i,j,k in cell.PxByPyByPz:
#                 cell.gridpoints[i,j,k].phiImported = None
        if importIndex != len(phiNew):
            print('Warning: import index not equal to len(phiNew)')
            print(importIndex)
            print(len(phiNew))
            
    def importDensityOnLeaves(self,rho):
        '''
        Import density values, apply to leaves
        '''

        importIndex = 0        
        for _,cell in self.masterList:
            if cell.leaf == True:
                for i,j,k in cell.PxByPyByPz:
                    gridpt = cell.gridpoints[i,j,k]
                    gridpt.rho = rho[importIndex]
                    importIndex += 1
                    

        if importIndex != len(rho):
            print('Warning: import index not equal to len(rho)')
            print(importIndex)
            print(len(rho))
            
    def importPhiNewOnLeaves(self,phiNew):
        '''
        Import phi difference values, apply to leaves
        '''
#         for _,cell in self.masterList:
#             for i,j,k in cell.PxByPyByPz:
#                 cell.gridpoints[i,j,k].phiImported = False
        importIndex = 0        
        for _,cell in self.masterList:
            if cell.leaf == True:
                for i,j,k in cell.PxByPyByPz:
                    gridpt = cell.gridpoints[i,j,k]
#                     if gridpt.phiImported == False:
                    gridpt.phiNew = phiNew[importIndex]
#                         gridpt.phiImported = True
                    importIndex += 1
                    
#         for _,cell in self.masterList:
#             for i,j,k in cell.PxByPyByPz:
#                 cell.gridpoints[i,j,k].phiImported = None
        if importIndex != len(phiNew):
            print('Warning: import index not equal to len(phiNew)')
            print(importIndex)
            print(len(phiNew))
            
    def importVhartreeOnLeaves(self,V_hartreeNew):
        '''
        Import V_coulomn values, apply to leaves
        '''

        importIndex = 0        
        for _,cell in self.masterList:
            if cell.leaf == True:
                for i,j,k in cell.PxByPyByPz:
                    gridpt = cell.gridpoints[i,j,k]
                    gridpt.v_hartree = V_hartreeNew[importIndex]
                    importIndex += 1

        if importIndex != len(V_hartreeNew):
            print('Warning: import index not equal to len(V_hartreeNew)')
            print(importIndex)
            print(len(V_hartreeNew))
            
    
                
    def copyPhiToFinalOrbital(self, n):
        # outdated
        for _,cell in self.masterList:
            for i,j,k in cell.PxByPyByPz:
                gridpt = cell.gridpoints[i,j,k]
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
        for _,cell in self.masterList:
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
#                 for i,j,k in cell.PxByPyByPz:
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
        rho = []
        
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
        phi10 = []
        phi11 = []
        phi12 = []
        phi13 = []
        phi14 = []
        phi15 = []
        phi16 = []
        phi17 = []
        phi18 = []
        phi19 = []
        phi20 = []
        phi21 = []
        phi22 = []
        phi23 = []
        phi24 = []
        phi25 = []
        phi26 = []
        phi27 = []
        phi28 = []
        phi29 = []
        for _,cell in self.masterList:
            if cell.leaf==True:
                for i,j,k in cell.PxByPyByPz:
                    gp = cell.gridpoints[i,j,k]
                    x.append(gp.x)
                    y.append(gp.y)
                    z.append(gp.z)
                    rho.append(gp.rho)
                    v.append(gp.v_eff)
                    phi0.append(gp.phi[0])
                    phi1.append(gp.phi[1])
                    phi2.append(gp.phi[2])
                    phi3.append(gp.phi[3])
                    phi4.append(gp.phi[4])
#                     phi5.append(gp.phi[5])
#                     phi6.append(gp.phi[6])
#                     phi7.append(gp.phi[7])
#                     phi8.append(gp.phi[8])
#                     phi9.append(gp.phi[9])
#                     phi10.append(gp.phi[10])
#                     phi11.append(gp.phi[11])
#                     phi12.append(gp.phi[12])
#                     phi13.append(gp.phi[13])
#                     phi14.append(gp.phi[14])
#                     phi15.append(gp.phi[15])
#                     phi16.append(gp.phi[16])
#                     phi17.append(gp.phi[17])
#                     phi18.append(gp.phi[18])
#                     phi19.append(gp.phi[19])
#                     phi20.append(gp.phi[20])
#                     phi21.append(gp.phi[21])
#                     phi22.append(gp.phi[22])
#                     phi23.append(gp.phi[23])
#                     phi24.append(gp.phi[24])
#                     phi25.append(gp.phi[25])
#                     phi26.append(gp.phi[26])
#                     phi27.append(gp.phi[27])
#                     phi28.append(gp.phi[28])
#                     phi29.append(gp.phi[29])
                    
                    
#         pointsToVTK(filename, np.array(x), np.array(y), np.array(z), data = 
#                     {"rho"  : np.array(rho),       "V" : np.array(v),  
#                     "Phi0"  : np.array(phi0),  "Phi1"  : np.array(phi1),  "Phi2"  : np.array(phi2),  "Phi3"  : np.array(phi3),  "Phi4"  : np.array(phi4),  
#                     "Phi5"  : np.array(phi5),  "Phi6"  : np.array(phi6),  "Phi7"  : np.array(phi7),  "Phi8"  : np.array(phi8),  "Phi9"  : np.array(phi9), 
#                     "Phi10" : np.array(phi10), "Phi11" : np.array(phi11), "Phi12" : np.array(phi12), "Phi13" : np.array(phi13), "Phi14" : np.array(phi14),  
#                     "Phi15" : np.array(phi15), "Phi16" : np.array(phi16), "Phi17" : np.array(phi17), "Phi18" : np.array(phi18), "Phi19" : np.array(phi19), 
#                     "Phi20" : np.array(phi20), "Phi21" : np.array(phi21), "Phi22" : np.array(phi22), "Phi23" : np.array(phi23), "Phi24" : np.array(phi24),  
#                     "Phi25" : np.array(phi25), "Phi26" : np.array(phi26), "Phi27" : np.array(phi27), "Phi28" : np.array(phi28), "Phi29" : np.array(phi29) } )
        
        # 7 wavefunctions (carbon monoxide)
#         pointsToVTK(filename, np.array(x), np.array(y), np.array(z), data = 
#                     {"rho" : np.array(rho), "V" : np.array(v),  "Phi0" : np.array(phi0), "Phi1" : np.array(phi1),#}) ,
#                     "Phi2" : np.array(phi2), "Phi3" : np.array(phi3), "Phi4" : np.array(phi4),#,
#                      "Phi5" : np.array(phi5), "Phi6" : np.array(phi6)  } )
        
        # 5 wavefunctions (oxygen)
        pointsToVTK(filename, np.array(x), np.array(y), np.array(z), data = 
                    {"rho" : np.array(rho), "V" : np.array(v),  "Phi0" : np.array(phi0), "Phi1" : np.array(phi1),
                    "Phi2" : np.array(phi2), "Phi3" : np.array(phi3), "Phi4" : np.array(phi4)  } )
        
        
#                      "Phi7" : np.array(phi7), "Phi8" : np.array(phi8), "Phi9" : np.array(phi9)})
        
        
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
        for _,cell in self.masterList:
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
        
    def interpolateDensity(self, xi,yi,zi, xf,yf,zf, numpts, plot=False, save=False):
        
        # generate linspace from endpoint to endpoint
        x = np.linspace(xi,xf,numpts)
        y = np.linspace(yi,yf,numpts)
        z = np.linspace(zi,zf,numpts)
        
        r = np.sqrt( (x-xi)**2 + (y-yi)**2 + (z-zi)**2 )
        rho = np.empty_like(r)
        
        # For each point, locate leaf cell that owns it
        for i in range(numpts):
            # point is at (x[i],y[i],z[i])
            cell = self.findOwner(x[i],y[i],z[i])
    
            # Construct density interpolator if necessary
            if not hasattr(Cell, "densityInterpolator"):
                cell.setDensityInterpolator()
            
            rho[i] = cell.densityInterpolator(x[i],y[i],z[i])
            
            # evaluate at point to get rho value
#         if referenceRho==True:
#             initialRho = np.zeros_like(r)
#             
#             for atom in self.atoms:
#                 dx = x-atom.x
#                 dy = y-atom.y
#                 dz = z-atom.z
#                 
#                 initialRho += atom.d
        if plot==True:
            fig = plt.figure()
            plt.semilogy(r,rho)
            plt.title('Density along line from (%1.2f, %1.2f, %1.2f) to (%1.2f, %1.2f, %1.2f)' %(xi,yi,zi,xf,yf,zf))
            if save==False:
                plt.show()
            else:
                print('Saving figure to ', save)
                plt.savefig(save+'.pdf',format='pdf',bbox_inches='tight')
                plt.close(fig)
        return r, rho
    
    def findOwner(self,x,y,z):
        # This could be optimized if needed.
        for _,cell in self.masterList:
            if cell.leaf==True:
                if ( (cell.xmin<=x) and (cell.xmax>=x) ):
                    if ( (cell.ymin<=y) and (cell.ymax>=y) ):
                        if ( (cell.zmin<=z) and (cell.zmax>=z) ):
                            return cell
                                    
def TestTreeForProfiling():
    xmin = ymin = zmin = -12
    xmax = ymax = zmax = -xmin
    tree = Tree(xmin,xmax,ymin,ymax,zmin,zmax)
    tree.buildTree( maxLevels=4, divideTolerance=0.07,printTreeProperties=True)
    
               
if __name__ == "__main__":
    TestTreeForProfiling()
    

    
    
       
    