'''
Created on Jan 22, 2018

@author: nathanvaughn
'''    
# import tools
import sys
sys.path.append('methods/')
import numpy as np
import time
import cProfile

# import classes
from mesh3D import generate_grid, normalize_wavefunction, Slice2D, Slice2D_Comparison
from hydrogen_potential import potential, trueEnergy, trueWavefunction
from convolution import conv
from hamiltonian import EnergyUpdate

print('main.py was launched')
print(__name__)
def run():
# print('name == main')
    start = time.time()
    nx = ny = nz = 12
    xmin = ymin = zmin = -6
    xmax = ymax = zmax = 6
    dx = (xmax-xmin)/nx
    dy = dz = dx
    t1 = time.time()
    x,y,z = generate_grid(nx,ny,nz,xmin,xmax,ymin,ymax,zmin,zmax)
    print('grid generation took ', time.time()-t1,' seconds.')
    t2 = time.time()
    V = potential(x, y, z)
    print('Evaluating potential took ', time.time()-t2,' seconds.')
    psi = np.random.rand(nx,ny,nz)
    psi_true = trueWavefunction(1, x, y, z)
    psi_true = normalize_wavefunction(psi_true, dx, dy, dz)
    energy_true = trueEnergy(1)
    E = -1
    
    deltaE = 1
    count=1
    while deltaE > 1e-4:
        print('Green Iteration Count: ', count)
        E_old = E
        t3 = time.time()
        psi = conv(V, E, psi, x, y, z)
        print('performing convolution took ', time.time()-t3,' seconds.')
        t4 = time.time()
        psi = normalize_wavefunction(psi, dx, dy, dz)
        print('normalization took ', time.time()-t4,' seconds.')
        t5 = time.time()
        E = EnergyUpdate(V, psi, x, y, z)
        print('updating energy took ', time.time()-t5,' seconds.')
        print('E = ',E)
        deltaE = abs(E-E_old)
        print('Residual: ',deltaE)
        print('Energy Error: ', energy_true-E,'\n')
        count+=1
        
    end = time.time()
    print('Grid dim: ',xmin,' to ', xmax)
    print('nx = ny = nz = ', nx)
    print('Final Energy error: ', energy_true-E)
    print('computation time: ', end-start)
        
    #     Slice2D_Comparison(psi,psi_true,x,y,int(nz/2))
        
if ( (__name__ == "__main__") or (__name__ == "main") ):
    run()
