import numpy as np
import sys
sys.path.insert(1, '/Users/nathanvaughn/Documents/GitHub/TAGI/3D-GreenIterations/src/utilities')
sys.path.insert(1, '/Users/nathanvaughn/Documents/GitHub/TAGI/3D-GreenIterations/src/dataStructures')
sys.path.insert(1, '/home/njvaughn/TAGI/3D-GreenIterations/src/utilities')
from mpiUtilities import global_dot, rprint
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


from scipy.special import sph_harm
 


def initializeOrbitalsRandomly(atoms,coreRepresentation,orbitals,nOrbitals,X,Y,Z):
    rprint(0, "INITIALIZING ORBITALS RANDOMLY")
#     orbitals = np.zeros(nOrbitals,len(X))
    R2=X*X+Y*Y+Z*Z
    R = np.sqrt(R2)
    for m in range(nOrbitals):
        orbitals[m,:] = np.exp(-R)*np.sin(m*R)
    return orbitals

    
    
def initializeOrbitalsFromAtomicDataExternally(atoms,coreRepresentation,orbitals,nOrbitals,X,Y,Z,W): 
        aufbauList = ['10',                                     # n+ell = 1
                      '20',                                     # n+ell = 2
                      '21', '30',                               # n+ell = 3
                      '31', '40', 
                      '32', '41', '50'
                      '42', '51', '60'
                      '43', '52', '61', '70']

        orbitalIndex=0
        initialOccupations=[]
        initialEnergies=[]
        
        for atom in atoms:
            nAtomicOrbitals = atom.nAtomicOrbitals
            electronCount=0
                
            
            
            rprint(rank,'Initializing orbitals for atom Z = %i (%i atomic orbitals) located at (x, y, z) = (%6.3f, %6.3f, %6.3f)' 
                      %(atom.atomicNumber, atom.nAtomicOrbitals, atom.x,atom.y,atom.z))
            rprint(rank,'Orbital index = %i'%orbitalIndex)            
            singleAtomOrbitalCount=0
            for nell in aufbauList:
                
                if singleAtomOrbitalCount< nAtomicOrbitals:  
                    n = int(nell[0])
                    ell = int(nell[1])
                    psiID = 'psi'+str(n)+str(ell)
#                     rprint(0, 'Using ', psiID)
                    # filling these requires 2 * (2*ell+1) electrons
                    if atom.nuclearCharge - electronCount - 2*(2*ell+1) >= 0:
                        occupation=2.0
                    else:
                        occupation = (atom.nuclearCharge - electronCount) / ((2*ell+1))
                    for m in range(-ell,ell+1):
                            
                        
                        if psiID in atom.interpolators:  # pseudopotentials don't start from 10, 20, 21,... they start from the valence, such as 30, 31, ...
                            
                            initialEnergies.append( -atom.nuclearCharge / (n+ell))  # provide initial eigenvalue guess.  Higher (n,ell) closer to zero.
                            initialOccupations.append(occupation)
                            electronCount+=occupation
                            rprint(rank, "Initial (n,ell,m)=(%i,%i,%i) wavefunction gets occupation %f" %(n,ell,m,occupation))
                            rprint(rank, "Accounted for %f of %i electrons now." %(electronCount,atom.nuclearCharge))
                            #input()
                            
                            dx = X-atom.x
                            dy = Y-atom.y
                            dz = Z-atom.z
                            phi = np.zeros(len(dx))
                            r = np.sqrt( dx**2 + dy**2 + dz**2 )
                            inclination = np.arccos(dz/r)
    #                         rprint(0, 'Type(dx): ', type(dx))
    #                         rprint(0, 'Type(dy): ', type(dy))
    #                         rprint(0, 'Shape(dx): ', np.shape(dx))
    #                         rprint(0, 'Shape(dy): ', np.shape(dy))
                            azimuthal = np.arctan2(dy,dx)
                            
                            if m<0:
                                Ysp = (sph_harm(m,ell,azimuthal,inclination) + (-1)**m * sph_harm(-m,ell,azimuthal,inclination))/np.sqrt(2) 
                            if m>0:
                                Ysp = 1j*(sph_harm(m,ell,azimuthal,inclination) - (-1)**m * sph_harm(-m,ell,azimuthal,inclination))/np.sqrt(2)
    #                                     if ( (m==0) and (ell>1) ):
                            if ( m==0 ):
                                Ysp = sph_harm(m,ell,azimuthal,inclination)
    #                                     if ( (m==0) and (ell<=1) ):
    #                                         Y = 1
                            if np.max( abs(np.imag(Ysp)) ) > 1e-14:
                                rprint(0, 'imag(Y) ', np.imag(Ysp))
                                return
    #                                     Y = np.real(sph_harm(m,ell,azimuthal,inclination))
    #                         phi = atom.interpolators[psiID](r)*np.real(Y)
                            try:
                                phi = atom.interpolators[psiID](r)*np.real(Ysp) / np.sqrt(4*np.pi*r*r) 
                            except ValueError:
                                phi = 0.0   # if outside the interpolation range, assume 0.
                            except KeyError:
                                for key, value in atom.interpolators.items() :
                                    rprint(rank,key, value)
                                exit(-1)
                            
                            
                            integralPhiSquared = global_dot(phi**2,W,comm)
                            phi /= np.sqrt(integralPhiSquared)
                            
                            orbitals[orbitalIndex,:] = np.copy(phi)
    #                         self.importPhiOnLeaves(phi, orbitalIndex)
    #                         self.normalizeOrbital(orbitalIndex)
                            
                            rprint(rank,'Orbital %i filled with (n,ell,m) = (%i,%i,%i) ' %(orbitalIndex,n,ell,m))
                            orbitalIndex += 1
                            singleAtomOrbitalCount += 1
                    
#                 else:
#                     n = int(nell[0])
#                     ell = int(nell[1])
#                     psiID = 'psi'+str(n)+str(ell)
#                     rprint(0, 'Not using ', psiID)
                        
        if orbitalIndex < nOrbitals:
            rprint(rank,"Didn't fill all the orbitals.  Should you initialize more?  Randomly, or using more single atom data?")
#             rprint(0, 'Filling extra orbitals with decaying exponential.')
            rprint(rank,'Filling extra orbitals with random initial data.')
            for ii in range(orbitalIndex, nOrbitals):
                R = np.sqrt(X*X+Y*Y+Z*Z)
#                 orbitals[:,ii] = np.exp(-R)*np.sin(R)
                orbitals[ii,:] = np.random.rand(len(R))
                initialOccupations.append(0.0)
                initialEnergies.append(-0.1)
#                 self.initializeOrbitalsRandomly(targetOrbital=ii)
#                 self.initializeOrbitalsToDecayingExponential(targetOrbital=ii)
#                 self.orthonormalizeOrbitals(targetOrbital=ii)
        if orbitalIndex > nOrbitals:
            rprint(rank,"Filled too many orbitals, somehow.  That should have thrown an error and never reached this point.")
                        

#         
#         for m in range(self.nOrbitals):
#             self.normalizeOrbital(m)

        rprint(rank, "initial occupations: ",initialOccupations)
        #input()
        return orbitals, initialOccupations, initialEnergies
    

def initializeDensityFromAtomicDataExternally(x,y,z,w,atoms,coreRepresentation):
        
    rho = np.zeros(len(x))
    ccrho = np.zeros(len(x))
    
    
    totalElectrons = 0
    for atom in atoms:
        
        r = np.sqrt( (x-atom.x)**2 + (y-atom.y)**2 + (z-atom.z)**2 )
        
        if coreRepresentation=="AllElectron":
            totalElectrons += atom.atomicNumber
            try:
                rho += atom.interpolators['density'](r)
            except ValueError:
                rho += 0.0   # if outside the interpolation range, assume 0.
        elif coreRepresentation=="Pseudopotential":
            totalElectrons += atom.PSP.psp['header']['z_valence']
            rho += atom.PSP.evaluateDensityInterpolator(r)
            
            if atom.psp['headers']['core_correction']==True:
                rprint(rank,"Initializing core charge density for atom ", atom)
                ccrho += atom.PSP.evaluateCoreChargeDensityInterpolator(r)

            else:
                rprint(rank,"Not initializing core charge density for atom ", atom)
            
            
            
        rprint(rank,"max density: ", max(abs(rho)))
        rprint(rank,"max core charge density: ", max(abs(ccrho)))
        rprint(rank,"cumulative number of electrons: ", totalElectrons)


#     rprint(0, "NOT NORMALIZING INITIAL DENSITY.")
    rho *= totalElectrons / global_dot(rho,w,comm)
    
    return rho, ccrho



