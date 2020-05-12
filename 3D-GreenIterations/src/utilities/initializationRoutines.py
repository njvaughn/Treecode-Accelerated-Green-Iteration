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




def initializeDensityFromAtomicDataExternally(x,y,z,w,atoms,coreRepresentation):
        
    rho = np.zeros(len(x))
    
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
            
        rprint(rank,"max density: ", max(abs(rho)))
        rprint(rank,"cumulative number of electrons: ", totalElectrons)


#     rprint(0, "NOT NORMALIZING INITIAL DENSITY.")
    rho *= totalElectrons / global_dot(rho,w,comm)
    
    return rho
