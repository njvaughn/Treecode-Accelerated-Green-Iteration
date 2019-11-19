'''
@author: nathanvaughn
'''
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.special import erf, sph_harm

import os
from PseudopotentialStruct import ONCV_PSP

from mpiUtilities import global_dot, rprint




class Atom(object):
    '''
    The gridpoint object.  Will contain the coordinates, wavefunction value, and any other metadata such as
    whether or not the wavefunction has been updated, which cells the gridpoint belongs to, etc.
    '''
    def __init__(self, x,y,z,atomicNumber,nAtomicOrbitals, coreRepresentation):
        '''
        Atom Constructor
        '''
        self.x = x
        self.y = y
        self.z = z
        self.atomicNumber = int(atomicNumber)
        self.orbitalInterpolators()
        self.nAtomicOrbitals = nAtomicOrbitals
        self.coreRepresentation = coreRepresentation
        
    def setPseudopotentialObject(self, PSPs,verbose=0):
        ## If the pseudopotential dictionary already contains this atomic number, have it point there.
        ## Otherwise, need to create a new one.
        print("Setting PSP for atomic number ", self.atomicNumber, self)
        try: 
            self.PSP = PSPs[str(self.atomicNumber)]
            if verbose>0: print("PSP already present for atomic number ", self.atomicNumber)
        except KeyError:
            if verbose>0: print("PSP not already present for atomic number ", self.atomicNumber)
            PSPs[str(self.atomicNumber)] = ONCV_PSP(self.atomicNumber)
            if verbose>0: print("Updated PSPs: ",PSPs)
            self.PSP = PSPs[str(self.atomicNumber)]
        
    def V_all_electron(self,x,y,z):
        r = np.sqrt((x - self.x)**2 + (y-self.y)**2 + (z-self.z)**2)
        return -self.atomicNumber/r
    
    def V_local_pseudopotential(self,x,y,z):
        r = np.sqrt((x - self.x)**2 + (y-self.y)**2 + (z-self.z)**2)
        return self.PSP.evaluateLocalPotentialInterpolator(r)
    
    def V_nonlocal_pseudopotential_times_psi(self,x,y,z,psi,W,comm):
        
        output = np.zeros(len(psi))     
        ## sum over the projectors, increment the nonloncal potential. 
        for i in range(self.numberOfChis):
            C = global_dot( psi, self.Chi[str(i)]*W, comm)
            output += C * self.Chi[str(i)] / self.Dion[str(i)]  # check how to use Dion.  Is it h, or is it 1/h?  Or something else?
        return output
    
    def generateChi(self,X,Y,Z):
        self.Chi = {}
        self.Dion = {}
        D_ion_array = np.array(self.PSP.psp['D_ion'][::self.PSP.psp['header']['number_of_proj']+1]) # grab diagonals of matrix
        num_ell = int(self.PSP.psp['header']['number_of_proj']/2)  # 2 projectors per ell for ONCV
        ID=0
        for ell in range(num_ell):
            D_ion = D_ion_array[ID%2]
            for p in [0,1]:  # two projectors per ell for ONCV
                
                for m in range(-ell,ell+1):

                    dx = X-self.x
                    dy = Y-self.y
                    dz = Z-self.z
                    chi = np.zeros(len(dx))
                    r = np.sqrt( dx**2 + dy**2 + dz**2 )
                    inclination = np.arccos(dz/r)
                    azimuthal = np.arctan2(dy,dx)

                    if m<0:
                        Ysp = (sph_harm(m,ell,azimuthal,inclination) + (-1)**m * sph_harm(-m,ell,azimuthal,inclination))/np.sqrt(2) 
                    if m>0:
                        Ysp = 1j*(sph_harm(m,ell,azimuthal,inclination) - (-1)**m * sph_harm(-m,ell,azimuthal,inclination))/np.sqrt(2)

                    if ( m==0 ):
                        Ysp = sph_harm(m,ell,azimuthal,inclination)

                    if np.max( abs(np.imag(Ysp)) ) > 1e-14:
                        print('imag(Y) ', np.imag(Ysp))
                        return

                    chi = self.PSP.evaluateProjectorInterpolator(2*ell+p, r)*np.real(Ysp)
                    self.Chi[str(ID)] = chi
                    self.Dion[str(ID)] = D_ion
                    ID+=1
        self.numberOfChis = ID  # this is larger than number of projectors, which don't depend on m
                    
    def V_pseudopotential_times_psi(self,x,y,z,psi,W,comm):
        ## Call the local and nonlocal pseudopotential calculations.
        return self.V_local_pseudopotential(x,y,z)*psi + self.V_nonlocal_pseudopotential_times_psi(x,y,z,psi,W,comm)
        
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
        
        
    def orbitalInterpolators(self,verbose=0):
        
#         print("Setting up interpolators.")
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
                self.interpolators[singleAtomData[:5]] = InterpolatedUnivariateSpline(data[:,0],data[:,1],k=3,ext=0)
            elif singleAtomData[:7]=='density':
                data = np.genfromtxt(path+singleAtomData)
                self.interpolators[singleAtomData[:7]] = InterpolatedUnivariateSpline(data[:,0],data[:,1],k=3,ext=0)        
