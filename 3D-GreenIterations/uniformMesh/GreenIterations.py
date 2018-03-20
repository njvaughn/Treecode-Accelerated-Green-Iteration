'''
Created on Jan 22, 2018

@author: nathanvaughn
'''    
# import tools
import sys
sys.path.append('methods/')
import numpy as np
import time
import socket


# import classes
from mesh3D import generate_grid, normalize_wavefunction, Slice2D, Slice2D_Comparison, trapezoid_weight_matrix
from hydrogen_potential import potential, trueEnergy, trueWavefunction, potential_smoothed
from convolution import conv
from hamiltonian import EnergyUpdate

print('main.py was launched')
print(__name__)
print(socket.gethostname())
def run(machine, numpts, boxsize, tolerance):
# print('name == main')
    print('tolerance: ', tolerance)
    start = time.time()
    nx = ny = nz = numpts
    xmin = ymin = zmin = -boxsize
    xmax = ymax = zmax = boxsize
    dx = (xmax-xmin)/nx
    dy = dz = dx
    x,y,z = generate_grid(nx,ny,nz,xmin,xmax,ymin,ymax,zmin,zmax)
    W = trapezoid_weight_matrix(nx, ny, nz)
#     V = potential(x, y, z)
    V = potential_smoothed(x, y, z, dx/2)
    psi = np.random.rand(nx,ny,nz)
    psi_true = trueWavefunction(1, x, y, z)
    psi_true = normalize_wavefunction(psi_true, dx, dy, dz, W)
    energy_true = trueEnergy(1)
    E = -1
    
    deltaE = 1
    count=1
    
#     vectorized_conv = np.vectorize(conv_for_vectorization)
#     itargets,jtargets,ktargets = np.meshgrid(np.arange(0,nx), np.arange(0,ny), np.arange(0,nz))
    while deltaE > tolerance:
        print('Green Iteration Count: ', count)
        E_old = E
        psi = conv(V, E, psi, x, y, z)
#         psi = vectorized_conv(V,E,psi,x,y,z)
#         psi = vectorized_conv(V,E,psi,x,y,z,itargets.flatten(),jtargets.flatten(),ktargets.flatten())
        psi = normalize_wavefunction(psi, dx, dy, dz, W)
        E = EnergyUpdate(V, psi, x, y, z, W)
        print('E = ',E)
        deltaE = abs(E-E_old)
        print('Residual: ',deltaE)
        wave_error = psi-psi_true
        print('Energy Error: ', energy_true-E)
        print('Linf wave error: ', np.max(np.abs(wave_error)))
        print('L2 wave error: ', np.sqrt(np.sum(wave_error*wave_error)*dx*dy*dz))
        print()
        count+=1
        
    end = time.time()
    wave_error = psi-psi_true
    print('Grid dim: ',xmin,' to ', xmax)
    print('nx = ny = nz = ', nx)
    print('Final Energy error: ', energy_true-E)
    print('Linf wave error: ', np.max(np.abs(wave_error)))
    print('L2 wave error: ', np.sqrt(np.sum(wave_error*wave_error)*dx*dy*dz))
    print('computation time: ', end-start)
    
    if machine == "Nathan-Vaughns-MacBook-Pro.local":
        Slice2D_Comparison(psi,psi_true,x,y,int(nz/2))
        Slice2D(psi-psi_true, x, y, int(nz/2))
        
        
    #     Slice2D_Comparison(psi,psi_true,x,y,int(nz/2))
        
if ( (__name__ == "__main__") or (__name__ == "GreenIterations") ):
    run(socket.gethostname(),int(sys.argv[1]),int(sys.argv[2]),float(sys.argv[3]))

