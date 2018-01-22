'''
Created on Jan 22, 2018

@author: nathanvaughn
'''    
# import tools
import numpy as np
from pympler import asizeof

# import classes
from mesh3D import generate_grid, normalize_wavefunction
from hydrogen_potential import potential, trueEnergy, trueWavefunction
from convolution import conv
from hamiltonian import EnergyUpdate


if __name__ == "__main__":
    nx = ny = nz = 10
    xmin = ymin = zmin = -6
    xmax = ymax = zmax = 6
    dx = (xmax-xmin)/nx
    dy = dz = dx
    x,y,z = generate_grid(nx,ny,nz,xmin,xmax,ymin,ymax,zmin,zmax)
    V = potential(x, y, z)
    psi = np.random.rand(nx,ny,nz)
    E = -1
    
    deltaE = 1
    count=1
    while deltaE > 1e-5:
        print('Green Iteration Count: ', count)
        E_old = E
        psi = conv(V, E, psi, x, y, z)
        psi = normalize_wavefunction(psi, dx, dy, dz)
        E = EnergyUpdate(V, psi, x, y, z)
        print('E = ',E)
        deltaE = abs(E-E_old)
        print('Residual: ',deltaE,'\n')
        count+=1
    

