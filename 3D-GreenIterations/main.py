'''
Created on Jan 22, 2018

@author: nathanvaughn
'''    
# import tools
import sys
sys.path.append('methods/')
import numpy as np
from pympler import asizeof
import time

# import classes
from mesh3D import generate_grid, normalize_wavefunction, Slice2D, Slice2D_Comparison
from hydrogen_potential import potential, trueEnergy, trueWavefunction
from convolution import conv
from hamiltonian import EnergyUpdate
from dask.array.random import normal


if __name__ == "__main__":
    start = time.time()
    nx = ny = nz = 6
    xmin = ymin = zmin = -6
    xmax = ymax = zmax = 6
    dx = (xmax-xmin)/nx
    dy = dz = dx
    x,y,z = generate_grid(nx,ny,nz,xmin,xmax,ymin,ymax,zmin,zmax)
    V = potential(x, y, z)
    psi = np.random.rand(nx,ny,nz)
    psi_true = trueWavefunction(1, x, y, z)
    psi_true = normalize_wavefunction(psi_true, dx, dy, dz)
    energy_true = trueEnergy(1)
    E = -1
    
    deltaE = 1
    count=1
    while deltaE > 1e-3:
        print('Green Iteration Count: ', count)
        E_old = E
        psi = conv(V, E, psi, x, y, z)
        psi = normalize_wavefunction(psi, dx, dy, dz)
        E = EnergyUpdate(V, psi, x, y, z)
        print('E = ',E)
        deltaE = abs(E-E_old)
        print('Residual: ',deltaE)
        print('Energy Error: ', energy_true-E,'\n')
        count+=1
        
    end = time.time()
    print('computation time: ', end-start)
#     Slice2D_Comparison(psi,psi_true,x,y,int(nz/2))
    

