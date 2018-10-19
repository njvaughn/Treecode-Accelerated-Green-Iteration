'''
@author: nathanvaughn
'''
import numpy as np
from scipy.interpolate import interp1d
import os

class Atom(object):
    '''
    The gridpoint object.  Will contain the coordinates, wavefunction value, and any other metadata such as
    whether or not the wavefunction has been updated, which cells the gridpoint belongs to, etc.
    '''
    def __init__(self, x,y,z,atomicNumber):
        '''
        Atom Constructor
        '''
        self.x = x
        self.y = y
        self.z = z
        self.atomicNumber = int(atomicNumber)
        self.orbitalInterpolators()
        self.setNumberOfOrbitalsToInitialize()
       
    def V(self,x,y,z,epsilon=0.0000001):
        r = np.sqrt( epsilon**2 + (x - self.x)**2 + (y-self.y)**2 + (z-self.z)**2)
        if r ==0.0:
            print('Warning, evaluating potential at singularity!')
            return 0.0
        return -self.atomicNumber/r
    
    def setNumberOfOrbitalsToInitialize(self):
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
        
        print('Atom with Z=%i will get %i atomic orbitals initialized.' %(self.atomicNumber, self.nAtomicOrbitals))
        
        
    def orbitalInterpolators(self):
        
        self.interpolators = {}
        # search for single atom data, either on local machine or on flux
        if os.path.isdir('/Users/nathanvaughn/AtomicData/allElectron/z'+str(int(self.atomicNumber))+'/singleAtomData/'):
            # working on local machine
            path = '/Users/nathanvaughn/AtomicData/allElectron/z'+str(int(self.atomicNumber))+'/singleAtomData/'
        elif os.path.isdir('/home/njvaughn/AtomicData/allElectron/z'+str(int(self.atomicNumber))+'/singleAtomData/'):
            # working on flux
            path = '/home/njvaughn/AtomicData/allElectron/z'+str(int(self.atomicNumber))+'/singleAtomData/'
#             print('Warning warning warning: using initial orbitals for Beryllium despite doing lithium calc.')
#             path = '/home/njvaughn/AtomicData/allElectron/z'+str(int(self.atomicNumber+1))+'/singleAtomData/'
        else:
            print('Could not find single atom data...')
            print('Checked in: /Users/nathanvaughn/AtomicData/allElectron/z'+str(int(self.atomicNumber))+'/singleAtomData/')
            print('Checked in: /home/njvaughn/AtomicData/allElectron/z'+str(int(self.atomicNumber))+'/singleAtomData/')
            
            
        print('Using single atom data from:')
        print(path)
        for singleAtomData in os.listdir(path): 
            if singleAtomData[:3]=='psi':
                data = np.genfromtxt(path+singleAtomData)
                self.interpolators[singleAtomData[:5]] = interp1d(data[:,0],data[:,1])
            elif singleAtomData[:7]=='density':
                data = np.genfromtxt(path+singleAtomData)
                self.interpolators[singleAtomData[:7]] = interp1d(data[:,0],data[:,1])
        

        
        
        
        
        
        
        
        
        
        
        
        
         
            
            
        