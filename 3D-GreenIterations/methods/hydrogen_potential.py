import numpy as np

  
def potential(x,y,z=0):
    # e = 1
    # 4pi*epsilon0 = 1
    r = np.sqrt(x**2 + y**2 + z**2)
    if r.any()==0:
        return 'ValueError: Dividing by zero when evaluating the 1/r potential.'
    return -1/r

def trueEnergy(n):
    return -1/(2*n*n)

def trueWavefunction(n, x,y,z):
    return 2*np.exp(- np.sqrt(x*x + y*y + z*z ))
    