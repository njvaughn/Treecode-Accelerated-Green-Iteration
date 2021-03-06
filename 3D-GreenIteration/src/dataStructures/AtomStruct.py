'''
AtomStruct.py

The data structure to store each atom.  
Contains local functions to evaluate atom-centric fields,
wavefunction interpolators, PSP projectors, etc. 
@author: nathanvaughn
'''
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.special import erf, sph_harm
import mpi4py.MPI as MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

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
        self.nuclearCharge = self.atomicNumber # default for all-electron.  Will get changed if pseudopotential is initialized
        if coreRepresentation=="AllElectron":  # for PSP, these get set up somewhere else... sorry
            self.orbitalInterpolators(coreRepresentation)
        
        self.nAtomicOrbitals = nAtomicOrbitals
        self.coreRepresentation = coreRepresentation
        
    def setPseudopotentialObject(self, PSPs,verbose=0):
        ## If the pseudopotential dictionary already contains this atomic number, have it point there.
        ## Otherwise, need to create a new one.
        rprint(rank, "Setting PSP for atomic number ", self.atomicNumber, self)
        try: 
            self.PSP = PSPs[str(self.atomicNumber)]
            if verbose>0: rprint(rank , "PSP already present for atomic number ", self.atomicNumber)
        except KeyError:
            if verbose>0: rprint(rank , "PSP not already present for atomic number ", self.atomicNumber)
            PSPs[str(self.atomicNumber)] = ONCV_PSP(self.atomicNumber)
            if verbose>0: rprint(rank , "Updated PSPs: ",PSPs)
            self.PSP = PSPs[str(self.atomicNumber)]
        self.nuclearCharge = self.PSP.psp['header']['z_valence']  # set the nuclear charge for the PSP.
        
    def V_all_electron(self,x,y,z):
        r = np.sqrt((x - self.x)**2 + (y-self.y)**2 + (z-self.z)**2)
        return -self.atomicNumber/r
    
    def V_local_pseudopotential(self,x,y,z):
        r = np.sqrt((x - self.x)**2 + (y-self.y)**2 + (z-self.z)**2)
        return self.PSP.evaluateLocalPotentialInterpolator(r)
    
    def V_nonlocal_pseudopotential_times_psi_COARSE_OR_FINE(self,psi,Weights,interpolatedPsi=None,comm=None, outputMesh="coarse"):
        

        W=None
        Wf=None
        if interpolatedPsi is not None:
            # using two meshes.  Weights and Chis should come from fine Mesh
            Wf=Weights
        elif interpolatedPsi is None:
            W=Weights
            
        
        output = np.zeros(len(psi))     
        ## sum over the projectors, increment the nonloncal potential.   Compute C on the fine mesh.
        for i in range(self.numberOfChis):
            if interpolatedPsi is not None:
                if comm==None: # just a local computation:
                    C = np.dot( interpolatedPsi, self.FineChi[str(i)]*Wf)
#                     rprint(rank , "Chi %i, C = %1.8e" %(i,C))
                else:
                    C = global_dot( interpolatedPsi, self.FineChi[str(i)]*Wf, comm)
#                     rprint(rank, "Chi %i, C = %1.8e" %(i,C))
            elif interpolatedPsi is None:
                if comm==None: # just a local computation:
                    C = np.dot( psi, self.Chi[str(i)]*W)
#                     rprint(rank , "Chi %i, C = %1.8e" %(i,C))
                else:
                    C = global_dot( psi, self.Chi[str(i)]*W, comm)
#                     rprint(rank, "Chi %i, C = %1.8e" %(i,C))
                    
                
            

            if outputMesh=="coarse":
                output += C * self.Chi[str(i)] * self.Dion[str(i)]##/np.sqrt(2)  # check how to use Dion.  Is it h, or is it 1/h?  Or something else?
            elif outputMesh=="fine":
                output += C * self.FineChi[str(i)] * self.Dion[str(i)]##/np.sqrt(2)  # check how to use Dion.  Is it h, or is it 1/h?  Or something else?
            else:
                rprint(rank, "What should outputMesh be in atom struct?")
                exit(-1)
#         rprint(rank , "Exiting after first call to V_nonlocal_pseudopotential_times_psi")
#         exit(-1)
        return output
    
    
    def V_nonlocal_pseudopotential_times_psi_coarse(self,psi,Weights,finePsi,fineWeights,comm=None):
        
        output = np.zeros(len(psi))     
        ## sum over the projectors, increment the nonloncal potential.   Compute C on the coarse mesh.
        for i in range(self.numberOfFineChis):
            if comm==None: # just a local computation:
                C = np.dot( finePsi, self.FineChi[str(i)]*fineWeights)
            else:
                C = global_dot( finePsi, self.FineChi[str(i)]*fineWeights, comm)
            
            output += C * self.Chi[str(i)] * self.Dion[str(i)]##/np.sqrt(2)  # check how to use Dion.  Is it h, or is it 1/h?  Or something else?
            
        return output
    
    def V_nonlocal_pseudopotential_times_psi_SingleMesh(self,psi,Weights,comm=None):
        
        output = np.zeros(len(psi))     
        ## sum over the projectors, increment the nonloncal potential.   Compute C on the coarse mesh.
        for i in range(self.numberOfFineChis):
            if comm==None: # just a local computation:
                C = np.dot( psi, self.Chi[str(i)]*Weights)
            else:
                C = global_dot( psi, self.Chi[str(i)]*Weights, comm)
            
            output += C * self.Chi[str(i)] * self.Dion[str(i)]##/np.sqrt(2)  # check how to use Dion.  Is it h, or is it 1/h?  Or something else?
            
        return output
    
    def V_nonlocal_pseudopotential_times_psi_fine(self,psi,Weights,comm=None):
        
        output = np.zeros(len(psi))     
        ## sum over the projectors, increment the nonloncal potential.   Compute C on the fine mesh.
        for i in range(self.numberOfFineChis):
            if comm==None: # just a local computation:
                C = np.dot( psi, self.FineChi[str(i)]*Weights)
            else:
                C = global_dot( psi, self.FineChi[str(i)]*Weights, comm)
            
            output += C * self.FineChi[str(i)] * self.Dion[str(i)]##/np.sqrt(2)  # check how to use Dion.  Is it h, or is it 1/h?  Or something else?
            
        return output
    
    def generateChi(self,X,Y,Z):
        self.Chi = {}
        self.Dion = {}
        D_ion_array = np.array(self.PSP.psp['D_ion'][::self.PSP.psp['header']['number_of_proj']+1]) # grab diagonals of matrix, Ry->Ha /2 already accounted for by upf_to_json
#         rprint(rank,"D_ion_array = ", D_ion_array)
        num_ell = int(self.PSP.psp['header']['number_of_proj']/2)  # 2 projectors per ell for ONCV
        if self.PSP.psp['header']['number_of_proj']==2:
            assert (num_ell==1), "ERROR IN num_ell"
        if self.PSP.psp['header']['number_of_proj']==4:
            assert (num_ell==2), "ERROR IN num_ell"
        if self.PSP.psp['header']['number_of_proj']==8:
            assert (num_ell==4) ,"ERROR IN num_ell"
        ID=0
        D_ion_count=0
        for ell in range(num_ell):
             
            for p in [0,1]:  # two projectors per ell for ONCV
                 
                D_ion = D_ion_array[D_ion_count]
                D_ion_count+=1
#                 rprint(rank, "D_ion = ", D_ion)
                 
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
                        rprint(rank, 'imag(Y) ', np.imag(Ysp))
                        return
 
                    chi = self.PSP.evaluateProjectorInterpolator(2*ell+p, r)*np.real(Ysp) # Is this order the same as the setup order?
                    self.Chi[str(ID)] = chi
                    self.Dion[str(ID)] = D_ion
                    ID+=1
        self.numberOfChis = ID  # this is larger than number of projectors, which don't depend on m
        
    def generateFineChi(self,X,Y,Z):
        self.FineChi = {}
#         self.Dion = {}
#         D_ion_array = np.array(self.PSP.psp['D_ion'][::self.PSP.psp['header']['number_of_proj']+1]) # grab diagonals of matrix, Ry->Ha /2 already accounted for by upf_to_json
#         rprint(rank,"D_ion_array = ", D_ion_array)
        num_ell = int(self.PSP.psp['header']['number_of_proj']/2)  # 2 projectors per ell for ONCV
        if self.PSP.psp['header']['number_of_proj']==2:
            assert (num_ell==1), "ERROR IN num_ell"
        if self.PSP.psp['header']['number_of_proj']==4:
            assert (num_ell==2), "ERROR IN num_ell"
        if self.PSP.psp['header']['number_of_proj']==8:
            assert (num_ell==4) ,"ERROR IN num_ell"
        ID=0
#         D_ion_count=0
        for ell in range(num_ell):
            
            for p in [0,1]:  # two projectors per ell for ONCV
                
#                 D_ion = D_ion_array[D_ion_count]
#                 D_ion_count+=1
#                 rprint(rank, "D_ion = ", D_ion)
                
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
                        rprint(rank, 'imag(Y) ', np.imag(Ysp))
                        return

                    chi = self.PSP.evaluateProjectorInterpolator(2*ell+p, r)*np.real(Ysp) # Is this order the same as the setup order?
                    self.FineChi[str(ID)] = chi
#                     self.Dion[str(ID)] = D_ion
                    ID+=1
        self.numberOfFineChis = ID  # this is larger than number of projectors, which don't depend on m
        
    def normalizeChi(self,W,comm=None):
        for i in range(self.numberOfChis):
            
            norm=np.sqrt( global_dot(W,self.Chi[str(i)]**2,comm) )
            self.Chi[str(i)]/=norm
            
    def integrateProjectors(self, W, comm=None):
        I=0
        for i in range(self.numberOfChis):
            if comm==None:
                I+=np.dot(W,self.Chi[str(i)])
            else:
                rprint(rank, "Are you sure you want to integral projectors with mpi comm?")
                exit(-1)
        return I
     
    def normalizeFineChi(self,Wf,comm=None):
        rprint(rank, "DID YOU MEAN TO NORMALIZE CHI? Exiting...")
        exit(-1)
        for i in range(self.numberOfChis):       
            norm=np.sqrt( global_dot(Wf,self.FineChi[str(i)]**2,comm) )
            self.FineChi[str(i)]/=norm
    
    def removeTempChi(self):
        self.Chi=None
        self.Dion=None
        self.numberOfChis=0
    
                    
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
            rprint(rank, 'Not ready for > 30 atomic number.  Revisit atom.setNumberOfOrbitalsToInitialize()')
        
        if verbose>0: rprint(rank, 'Atom with Z=%i will get %i atomic orbitals initialized.' %(self.atomicNumber, self.nAtomicOrbitals))
        
        
    def orbitalInterpolators(self,coreRepresentation,verbose=0):
        
        if coreRepresentation=="AllElectron":
            atomDir="allElectron"
        elif coreRepresentation=="Pseudopotential":
            atomDir="pseudoPotential"
        else:
            rprint(rank, "What is coreRepresentation?  From orbitalInterpolators")
            exit(-1)
        
        
        if coreRepresentation=="AllElectron":
    #         rprint(rank, "Setting up interpolators.")
            self.interpolators = {}
            # search for single atom data, either on local machine or on flux
    #         if os.path.isdir('/Users/nathanvaughn/AtomicData/allElectron/z'+str(int(self.atomicNumber))+'/singleAtomData/'):
            if os.path.isdir('/Users/nathanvaughn/AtomicData/' + atomDir +'/z'+str(int(self.atomicNumber))+'/singleAtomData/'):
                # working on local machine
    #             path = '/Users/nathanvaughn/AtomicData/allElectron/z'+str(int(self.atomicNumber))+'/singleAtomData/'
                path = '/Users/nathanvaughn/AtomicData/' + atomDir +'/z'+str(int(self.atomicNumber))+'/singleAtomData/'
    #         elif os.path.isdir('/home/njvaughn/AtomicData/allElectron/z'+str(int(self.atomicNumber))+'/singleAtomData/'):
            elif os.path.isdir('/home/njvaughn/AtomicData/'+atomDir+'/z'+str(int(self.atomicNumber))+'/singleAtomData/'):
                # working on Flux or Great Lakes
    #             path = '/home/njvaughn/AtomicData/allElectron/z'+str(int(self.atomicNumber))+'/singleAtomData/'
                path = '/home/njvaughn/AtomicData/'+atomDir+'/z'+str(int(self.atomicNumber))+'/singleAtomData/'
            else:
                rprint(rank, 'Could not find single atom data...')
    #             rprint(rank, 'Checked in: /Users/nathanvaughn/AtomicData/allElectron/z'+str(int(self.atomicNumber))+'/singleAtomData/')
                rprint(rank, 'Checked in: /Users/nathanvaughn/AtomicData/'+atomDir+'/z'+str(int(self.atomicNumber))+'/singleAtomData/')
                rprint(rank, 'Checked in: /home/njvaughn/AtomicData/'+atomDir+'/z'+str(int(self.atomicNumber))+'/singleAtomData/')
                
                
            if verbose>0: rprint(rank, 'Using single atom data from:')
            if verbose>0: rprint(rank, path)
            for singleAtomData in os.listdir(path): 
                if singleAtomData[:3]=='psi':
                    data = np.genfromtxt(path+singleAtomData)
                    self.interpolators[singleAtomData[:5]] = InterpolatedUnivariateSpline(data[:,0],data[:,1],k=3,ext='zeros')
                elif singleAtomData[:7]=='density':
                    data = np.genfromtxt(path+singleAtomData)
                    self.interpolators[singleAtomData[:7]] = InterpolatedUnivariateSpline(data[:,0],data[:,1],k=3,ext='zeros')        
        
        
        elif coreRepresentation=="Pseudopotential":
            self.interpolators = {}
            if verbose>0: rprint(rank, 'Setting up orbital interpolators from PSP file')
            
            
            valenceShellStart=1
            coreCharge = self.atomicNumber - self.PSP.psp['header']['z_valence']
            if coreCharge>=2:
                valenceShellStart=2
            if coreCharge>=10:
                valenceShellStart=3
            if coreCharge>=18:
                valenceShellStart=4
            if coreCharge>=36:
                valenceShellStart=5
            if coreCharge>=54:
                rprint(rank,"AtomStruct needs to be updated to initialize atoms for n>=6.")
                return
            
            n=valenceShellStart-1
            for i in range(len(self.PSP.psp['atomic_wave_functions'])):
                ell = self.PSP.psp['atomic_wave_functions'][i]["angular_momentum"]
                if ell==0:
                    n += 1
                name = "psi"+str(n)+str(ell)
                radial_data = self.PSP.psp['atomic_wave_functions'][i]["radial_function"]
                radial_grid = self.PSP.psp['radial_grid']
                self.interpolators[name] = InterpolatedUnivariateSpline(radial_grid,radial_data,k=3,ext='zeros')
                
                rprint(rank,"Set pseudopotential orbital interpolator for ", name)
            
            