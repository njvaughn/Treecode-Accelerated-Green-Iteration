import numpy as np
from scipy.interpolate import RegularGridInterpolator
from hydrogenPotential import potential



class cell(object):
        
    def __init__(self, x,y,z,psi):
        self.psi = psi
        self.x = x
        self.y = y
        self.z = z
        self.dx = self.x[1]-self.x[0]
        self.dy = self.y[1]-self.y[0]
        self.dz = self.z[1]-self.z[0]
        self.volume = 2*self.dx * 2*self.dy * 2*self.dz
    
    def gradient_psi(self):
        self.grad = np.gradient(self.psi,self.dx,self.dy,self.dz,edge_order=2)        
 
    def interpolate_for_division(self):
        self.interpolator = RegularGridInterpolator((self.x, self.y, self.z), self.psi)
  
    def divide(self):
        children = np.empty((2,2,2), dtype=object)
        self.interpolate_for_division()
        xf = np.linspace(self.x[0],self.x[2],5)
        yf = np.linspace(self.y[0],self.y[2],5)
        zf = np.linspace(self.z[0],self.z[2],5)
        
        xm,ym,zm = np.meshgrid(xf,yf,zf, indexing='ij')
        psi_fine = self.interpolator((xm,ym,zm))
        
        # generate 8 children
        for ichild in range(2):
            for jchild in range(2):
                for kchild in range(2):
                    children[ichild,jchild,kchild] = cell( xf[2*ichild:2*ichild+3], 
                                                    yf[2*jchild:2*jchild+3], zf[2*kchild:2*kchild+3],
                                                    psi_fine[2*ichild:2*ichild+3, 2*jchild:2*jchild+3,2*kchild:2*kchild+3])
        return children

    def checkDivide(self,variationThreshold):
#         try:
#             self.grad
#         except AttributeError:
#             self.gradient_psi()
        
#         if np.max(np.abs(self.grad)) > gradientThreshold:
#         if np.max( [np.abs(self.grad[0][1,1,1]),np.abs(self.grad[1][1,1,1]),np.abs(self.grad[2][1,1,1])] ) > gradientThreshold:
#             self.NeedsDividing = True
#         else:
#             self.NeedsDividing = False
        variation = -potential(self.x[1],self.y[1],self.z[1])*(np.max(self.psi)**2 - np.min(self.psi)**2)*self.volume
        if variation > variationThreshold:
#         if ( np.max(self.psi) - np.min(self.psi)) > variationThreshold:
#         if ( np.max(self.psi) - np.min(self.psi))/np.max(self.psi) > variationThreshold:
            self.NeedsDividing = True
        else:
            self.NeedsDividing = False

    def evaluateKinetic_MidpointMethod(self):
#         try:
#             self.grad
#         except AttributeError:
#             self.gradient_psi()
        self.gradient_psi()
        
        Dxx = np.gradient(self.grad[0],self.dx,edge_order=2,axis=0)
        Dyy = np.gradient(self.grad[1],self.dy,edge_order=2,axis=1)
        Dzz = np.gradient(self.grad[2],self.dz,edge_order=2,axis=2)
        self.Laplacian = (Dxx + Dyy + Dzz)  # only use the Laplacian at the midpoint, for now at least
        self.Kinetic = -1/2*self.psi[1,1,1]*self.Laplacian[1,1,1]*self.volume
    
    def evaluatePotential_MidpointMethod(self):
#         r = np.sqrt(self.x[1]**2 + self.y[1]**2 + self.z[1]**2)
        self.Potential = self.psi[1,1,1]*potential(self.x[1],self.y[1],self.z[1])*self.psi[1,1,1]*self.volume
        
    def SimpsonWeights1D(self,nx): # Simpson weights, N needs to be odd
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
    
    def simpson_weight_matrix(self,nx, ny, nz):
        if ( (nx != ny) or (nx != nz) ):
            print('warning: simpson weights meant for uniform grid') 
        W1 = self.SimpsonWeights1D(nx)/3
        W2 = np.multiply.outer(W1,W1)
        W3 = np.multiply.outer(W2,W1) 
        return W3
        
    def evaluatePotential_SimpsonMethod(self):
#         r = np.sqrt(self.x**2 + self.y**2 + self.z**2)
        W3 = self.simpson_weight_matrix(3,3,3)
        self.Potential = np.sum(W3*self.psi*potential(self.x,self.y,self.z)*self.psi)*self.volume
        
    
    
    
    
    
            
    
    
    
    

    

    
    
    
    
        
    
    



