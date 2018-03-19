import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def generate_grid(nx, ny, nz, xmin, xmax, ymin, ymax, zmin, zmax):
#     if ( (xmax == -xmin) and (nx%2 != 0) ):
#         print('Mesh has a point at (0,0,0), is that okay?')
#         return
    return np.meshgrid( np.linspace(xmin,xmax,nx),np.linspace(ymin,ymax,ny),np.linspace(zmin,zmax,nz),indexing='ij')

def SimpsonWeights1D(nx): # Simpson weights, N needs to be odd
    if nx%2 == 0:
        print('error, even number of nodes')
        return
    wvec = np.zeros(nx)
    for i in range(nx):
        if i == 0:
            wvec[i] = 1
        elif i == nx-1:
            wvec[i] = 1
        elif i%2 != 0:
            wvec[i] = 4
        elif i%2 == 0:
            wvec[i] = 2
    return wvec

def TrapWeights1D(nx): # Simpson weights, N needs to be odd
    if nx%2 == 0:
        print('error, even number of nodes')
        return
    wvec = np.ones(nx)
    for i in range(nx):
        if i == 0:
            wvec[i] = 1/2
        elif i == nx-1:
            wvec[i] = 1/2
    return wvec

   
def midpoint_weight_matrix(nx, ny, nz):
    W3 = np.ones((nx,ny,nz)) 
    return W3

def simpson_weight_matrix(nx, ny, nz):
    if ( (nx != ny) or (nx != nz) ):
        print('warning: simpson weights meant for uniform grid') 
    W1 = SimpsonWeights1D(nx)/3
    W2 = np.multiply.outer(W1,W1)
    W3 = np.multiply.outer(W2,W1) 
    return W3

def trapezoid_weight_matrix(nx,ny,nz):
    if ( (nx != ny) or (nx != nz) ):
        print('warning: simpson weights meant for uniform grid') 
    W1 = TrapWeights1D(nx)
    W2 = np.multiply.outer(W1,W1)
    W3 = np.multiply.outer(W2,W1) 
    return W3
    
def normalize_wavefunction(psi,dx,dy,dz,W):
#     W is the weight matrix, corresponding to either trapezoid or simpson
    B = np.sum(W*psi*psi)*dx*dy*dz  # int psi^2 dxdydz 
    return psi/np.sqrt(B)

def Slice2D(psi,x,y,kindex):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(x[:,:,kindex], y[:,:,kindex], psi[:,:,kindex], cmap=cm.coolwarm,
                       linewidth=0, antialiased=True)
        plt.title('Wavefunction Error')
        plt.show()
        
def Slice2D_Comparison(psi,psi_true,x,y,kindex):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(x[:,:,kindex], y[:,:,kindex], psi[:,:,kindex], cmap=cm.coolwarm,
                       linewidth=0, antialiased=True)
        plt.title('Computed wavefunction')
        
        fig2 = plt.figure()
        ax2 = fig2.gca(projection='3d')
        surf2 = ax2.plot_surface(x[:,:,kindex], y[:,:,kindex], psi_true[:,:,kindex], cmap=cm.coolwarm,
                       linewidth=0, antialiased=True)
        plt.title('Analytic wavefunction')
        plt.show()
    



