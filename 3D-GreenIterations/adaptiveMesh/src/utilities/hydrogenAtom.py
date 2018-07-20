<<<<<<< HEAD
"""
Utility functions for the hydrogen atom.  The potential is given analytically, as opposed to 
tabulated.  The analytic energy values and wavefunctions are known for the single electron 
Schrodinger equation.  -- 03/20/2018 NV
"""

import numpy as np

def potential(x,y,z,epsilon=0.0):
    if epsilon != 0.0:
        print('Vepsilon != 0.0')
    r = np.sqrt(x*x+y*y+z*z + epsilon*epsilon)
    if r.any()==0:
        return 0
#         return 'ValueError: Dividing by zero when evaluating the -1/r potential.'
    return -1/r


def trueEnergy(n):
    m = n+1  # ground state is n=0.  
    return -1/(2*m*m)

def trueWavefunction(n, x,y,z):
    m = n+1 # ground state is n=0.
    if m>2:
        print('warning, don\'t have higher energy wavefunctions yet.  Only n=1')
    
    r = np.sqrt(x*x + y*y + z*z )
    if m == 1: return np.exp(- r)/np.sqrt(np.pi)
    if m == 2: return (2-r) * np.exp(-r/2)/(4*np.sqrt(2*np.pi))


=======
"""
Utility functions for the hydrogen atom.  The potential is given analytically, as opposed to 
tabulated.  The analytic energy values and wavefunctions are known for the single electron 
Schrodinger equation.  -- 03/20/2018 NV
"""

import numpy as np

def potential(x,y,z,epsilon=0.0):
    if epsilon != 0.0:
        print('Vepsilon != 0.0')
    r = np.sqrt(x*x+y*y+z*z + epsilon*epsilon)
    if r.any()==0:
        return 0
#         return 'ValueError: Dividing by zero when evaluating the -1/r potential.'
    return -1/r


def trueEnergy(n):
    m = n+1  # ground state is n=0.  
    return -1/(2*m*m)

def trueWavefunction(n, x,y,z):
    m = n+1 # ground state is n=0.
    if m>2:
        print('warning, don\'t have higher energy wavefunctions yet.  Only n=1')
    
    r = np.sqrt(x*x + y*y + z*z )
    if m == 1: return np.exp(- r)/np.sqrt(np.pi)
    if m == 2: return (2-r) * np.exp(-r/2)/(4*np.sqrt(2*np.pi))


>>>>>>> refs/remotes/eclipse_auto/master
