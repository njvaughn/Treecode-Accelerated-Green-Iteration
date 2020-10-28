'''
Routines for computing the fractional occupations according to Fermi-Dirac distributions
'''
import numpy as np
from scipy.optimize import broyden1, anderson, brentq


def fermiObjectiveFunction(orbitalEnergies, nElectrons, fermiEnergy, sigma):
            exponentialArg = (orbitalEnergies-fermiEnergy)/sigma
            temp = 1/(1+np.exp( exponentialArg ) )
            return nElectrons - 2 * np.sum(temp)
        
        
def computeOccupations(orbitalEnergies, nElectrons, Temperature):
    

    KB = 1/315774.6
    sigma = Temperature*KB
    
    
    
    fermiEnergy = brentq(fermiObjectiveFunction, orbitalEnergies[0], 1, xtol=1e-14)
    print('Fermi energy: ', fermiEnergy)
    exponentialArg = (orbitalEnergies-fermiEnergy)/sigma
    occupations = 2*1/(1+np.exp( exponentialArg ) )  # these are # of electrons, not fractional occupancy.  Hence the 2*
    print('Occupations: ', occupations)
    
    return occupations