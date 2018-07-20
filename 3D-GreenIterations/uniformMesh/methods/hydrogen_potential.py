<<<<<<< HEAD
import numpy as np

  
def potential(x,y,z=0):
    # e = 1
    # 4pi*epsilon0 = 1
    r = np.sqrt(x**2 + y**2 + z**2)
    if r.any()==0:
        return 'ValueError: Dividing by zero when evaluating the 1/r potential.'
    return -1/r

def potential_smoothed(x,y,z=0,epsilon=0):
    r = np.sqrt(x**2 + y**2 + z**2 + epsilon**2)
    if r.any()==0:
        return 'ValueError: Dividing by zero when evaluating the 1/r potential.'
    return -1/r

def trueEnergy(n):
    return -1/(2*n*n)

def trueWavefunction(n, x,y,z):
    if n!=1:
        print('warning, don\'t have higher energy wavefunctions yet.  Only n=1')
    return 2*np.exp(- np.sqrt(x*x + y*y + z*z ))
=======
import numpy as np

  
def potential(x,y,z=0):
    # e = 1
    # 4pi*epsilon0 = 1
    r = np.sqrt(x**2 + y**2 + z**2)
    if r.any()==0:
        return 'ValueError: Dividing by zero when evaluating the 1/r potential.'
    return -1/r

def potential_smoothed(x,y,z=0,epsilon=0):
    r = np.sqrt(x**2 + y**2 + z**2 + epsilon**2)
    if r.any()==0:
        return 'ValueError: Dividing by zero when evaluating the 1/r potential.'
    return -1/r

def trueEnergy(n):
    return -1/(2*n*n)

def trueWavefunction(n, x,y,z):
    if n!=1:
        print('warning, don\'t have higher energy wavefunctions yet.  Only n=1')
    return 2*np.exp(- np.sqrt(x*x + y*y + z*z ))
>>>>>>> refs/remotes/eclipse_auto/master
    