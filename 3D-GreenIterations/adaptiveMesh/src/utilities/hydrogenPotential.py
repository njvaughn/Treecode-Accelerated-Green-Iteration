import numpy as np

def potential(x,y,z,epsilon=0):
    r = np.sqrt(x*x+y*y+z*z + epsilon*epsilon)
    if r.any()==0:
        return 0
#         return 'ValueError: Dividing by zero when evaluating the -1/r potential.'
    return -1/r


def trueEnergy(n):
    return -1/(2*n*n)

def trueWavefunction(n, x,y,z):
    if n!=1:
        print('warning, don\'t have higher energy wavefunctions yet.  Only n=1')
    return np.exp(- np.sqrt(x*x + y*y + z*z ))/np.sqrt(np.pi)


