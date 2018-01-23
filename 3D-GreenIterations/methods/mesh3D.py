import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def generate_grid(nx, ny, nz, xmin, xmax, ymin, ymax, zmin, zmax):
    return np.meshgrid( np.linspace(xmin,xmax,nx),np.linspace(ymin,ymax,ny),np.linspace(zmin,zmax,nz),indexing='ij')
    
def normalize_wavefunction(psi,dx,dy,dz):
    B = np.sum(psi*psi)*dx*dy*dz  # int psi^2 dxdydz 
    return psi/np.sqrt(B)

def Slice2D(psi,x,y,kindex):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(x[:,:,kindex], y[:,:,kindex], psi[:,:,kindex], cmap=cm.coolwarm,
                       linewidth=0, antialiased=True)
        plt.show()
        
def Slice2D_Comparison(psi,psi_true,x,y,kindex):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(x[:,:,kindex], y[:,:,kindex], psi[:,:,kindex], cmap=cm.coolwarm,
                       linewidth=0, antialiased=True)
        
        fig2 = plt.figure()
        ax2 = fig2.gca(projection='3d')
        surf2 = ax2.plot_surface(x[:,:,kindex], y[:,:,kindex], psi_true[:,:,kindex], cmap=cm.coolwarm,
                       linewidth=0, antialiased=True)
        plt.show()
    