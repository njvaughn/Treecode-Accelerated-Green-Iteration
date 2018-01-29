'''
Created on Jan 22, 2018

@author: nathanvaughn
'''
import numpy as np

def Hamiltonian(V,psi,x,y,z):
    # hbar = 1
    # mass = 1
    delta_psi = Delta(psi,x,y,z)
    return -delta_psi/2 + V*psi

 
def Delta_old(psi,x,y,z):
    delta_x = x[1,0,0] - x[0,0,0]
    delta_y = y[0,1,0] - y[0,0,0]
    delta_z = z[0,0,1] - z[0,0,0]
    if ( ((delta_x - delta_y) > 1e-12) or ( (delta_x - delta_z > 1e-12)) ):
        return ('Second derivative is not prepared to handle different step sizes in different coordinates.  Need dx=dy=dz.')
    xp = np.roll(psi,1,axis=0)  # these rolls are using periodic BC...
    xm = np.roll(psi,-1,axis=0)
    yp = np.roll(psi,1,axis=1)
    ym = np.roll(psi,-1,axis=1)
    zp = np.roll(psi,1,axis=2)
    zm = np.roll(psi,-1,axis=2)
     
    delta_psi = (-6*psi + xp + xm + yp + ym + zp + zm)/(delta_x*delta_x)
    return delta_psi


def Delta(psi,x,y,z):

    dx = x[1,0,0] - x[0,0,0]
    dy = y[0,1,0] - y[0,0,0]
    dz = z[0,0,1] - z[0,0,0]
    
    psi_x,psi_y,psi_z = np.gradient(psi,dx,dy,dz,edge_order=2) 
    
    psi_xx = np.gradient(psi_x,dx,edge_order=2,axis=0)
    psi_yy = np.gradient(psi_y,dy,edge_order=2,axis=1)
    psi_zz = np.gradient(psi_z,dz,edge_order=2,axis=2)
    
    return psi_xx+psi_yy+psi_zz

def EnergyUpdate(V,psi,x,y,z):
    dx = x[1,0,0] - x[0,0,0]
    dy = y[0,1,0] - y[0,0,0]
    dz = z[0,0,1] - z[0,0,0]
    H = Hamiltonian(V,psi,x,y,z)
    return np.sum(psi*H)*dx*dy*dz  # the data is stored in 3D arrays, but this is just the 1D dot product that I want.  Sum of pointwise multiplications
