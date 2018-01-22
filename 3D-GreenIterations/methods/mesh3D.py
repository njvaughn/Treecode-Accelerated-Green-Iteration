import numpy as np

def generate_grid(nx, ny, nz, xmin, xmax, ymin, ymax, zmin, zmax):
    return np.meshgrid( np.linspace(xmin,xmax,nx),np.linspace(ymin,ymax,ny),np.linspace(zmin,zmax,nz),indexing='ij')
    
def normalize_wavefunction(psi,dx,dy,dz):
    B = np.sum(psi*psi)*dx*dy*dz  # int psi^2 dxdydz 
    return psi/np.sqrt(B)
    