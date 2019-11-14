'''
@author: nathanvaughn
'''
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.special import erf
import os




class Atom(object):
    '''
    The gridpoint object.  Will contain the coordinates, wavefunction value, and any other metadata such as
    whether or not the wavefunction has been updated, which cells the gridpoint belongs to, etc.
    '''
    def __init__(self, x,y,z,atomicNumber,nAtomicOrbitals):
        '''
        Atom Constructor
        '''
        self.x = x
        self.y = y
        self.z = z
        self.atomicNumber = int(atomicNumber)
        self.orbitalInterpolators()
        self.nAtomicOrbitals = nAtomicOrbitals
#         self.setNumberOfOrbitalsToInitialize()
        print("Set up atom with z=", atomicNumber)
     
       
    def V(self,x,y,z):
        r = np.sqrt((x - self.x)**2 + (y-self.y)**2 + (z-self.z)**2)
        return -self.atomicNumber/r
        
    
    def setNumberOfOrbitalsToInitialize(self,verbose=0):
        if self.atomicNumber <=2:       
            self.nAtomicOrbitals = 1    # 1S 
        elif self.atomicNumber <=4:     
            self.nAtomicOrbitals = 2    # 1S 2S 
        elif self.atomicNumber <=10:    
            self.nAtomicOrbitals = 5    # 1S 2S 2P
        elif self.atomicNumber <=12:
            self.nAtomicOrbitals = 6    # 1S 2S 2P 3S 
        elif self.atomicNumber <=18:
            self.nAtomicOrbitals = 9    # 1S 2S 2P 3S 3P
        elif self.atomicNumber <=20:
            self.nAtomicOrbitals = 10   # 1S 2S 2P 3S 3P 4S
        elif self.atomicNumber <=30:
            self.nAtomicOrbitals = 15   # 1S 2S 2P 3S 3P 4S 3D
        else:
            print('Not ready for > 30 atomic number.  Revisit atom.setNumberOfOrbitalsToInitialize()')
        
        if verbose>0: print('Atom with Z=%i will get %i atomic orbitals initialized.' %(self.atomicNumber, self.nAtomicOrbitals))
        
        
    def orbitalInterpolators(self,verbose=1):
        
        print("Setting up interpolators.")
        self.interpolators = {}
        # search for single atom data, either on local machine or on flux
        if os.path.isdir('/Users/nathanvaughn/AtomicData/allElectron/z'+str(int(self.atomicNumber))+'/singleAtomData/'):
            # working on local machine
            path = '/Users/nathanvaughn/AtomicData/allElectron/z'+str(int(self.atomicNumber))+'/singleAtomData/'
        elif os.path.isdir('/home/njvaughn/AtomicData/allElectron/z'+str(int(self.atomicNumber))+'/singleAtomData/'):
            # working on Flux or Great Lakes
            path = '/home/njvaughn/AtomicData/allElectron/z'+str(int(self.atomicNumber))+'/singleAtomData/'
        else:
            print('Could not find single atom data...')
            print('Checked in: /Users/nathanvaughn/AtomicData/allElectron/z'+str(int(self.atomicNumber))+'/singleAtomData/')
            print('Checked in: /home/njvaughn/AtomicData/allElectron/z'+str(int(self.atomicNumber))+'/singleAtomData/')
            
            
        if verbose>0: print('Using single atom data from:')
        if verbose>0: print(path)
        for singleAtomData in os.listdir(path): 
            if singleAtomData[:3]=='psi':
                data = np.genfromtxt(path+singleAtomData)
                print(singleAtomData[:5])
                print(data[0,0], data[-1,0])
                print(data[0,1], data[-1,1],"\n")
                self.interpolators[singleAtomData[:5]] = InterpolatedUnivariateSpline(data[:,0],data[:,1],k=3,ext=0)
            elif singleAtomData[:7]=='density':
                data = np.genfromtxt(path+singleAtomData)
                self.interpolators[singleAtomData[:7]] = InterpolatedUnivariateSpline(data[:,0],data[:,1],k=3,ext=0)
#                 self.interpolators[singleAtomData[:7]] = InterpolatedUnivariateSpline(data[:,0],data[:,1],k=3,ext='const')
        

