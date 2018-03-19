import numpy as np
from scipy.interpolate import RegularGridInterpolator
from hydrogenPotential import potential, smoothedPotential



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
        children = np.empty((2,2,2), dtype=object) # create array to hold children
        self.interpolate_for_division() # generate the local interpolator
        xf = np.linspace(self.x[0],self.x[2],5) # generate the fine mesh
        yf = np.linspace(self.y[0],self.y[2],5)
        zf = np.linspace(self.z[0],self.z[2],5)
        
        xm,ym,zm = np.meshgrid(xf,yf,zf, indexing='ij')
        psi_fine = self.interpolator((xm,ym,zm)) # interpolate psi onto the fine mesh
        
        # generate 8 children
        for ichild in range(2):
            for jchild in range(2):
                for kchild in range(2):
                    children[ichild,jchild,kchild] = cell( xf[2*ichild:2*ichild+3], 
                                                    yf[2*jchild:2*jchild+3], zf[2*kchild:2*kchild+3],
                                                    psi_fine[2*ichild:2*ichild+3, 2*jchild:2*jchild+3,2*kchild:2*kchild+3])
        return children

    def checkDivide(self,variationThreshold):
        variation = (np.max(self.psi) - np.min(self.psi))

        if variation > variationThreshold:
            self.NeedsDividing = True
        else:
            self.NeedsDividing = False

    def evaluateKinetic(self,W):
        try:
            self.grad
        except AttributeError:
            self.gradient_psi()
        Dxx = np.gradient(self.grad[0],self.dx,edge_order=2,axis=0)
        Dyy = np.gradient(self.grad[1],self.dy,edge_order=2,axis=1)
        Dzz = np.gradient(self.grad[2],self.dz,edge_order=2,axis=2)
        Laplacian = (Dxx + Dyy + Dzz)  # only use the Laplacian at the midpoint, for now at least
        self.Kinetic = -1/2*np.sum( W*self.psi*Laplacian*self.volume )
        
    def evaluatePotential(self,W,epsilon):
        self.Potential = np.sum(W*self.psi*smoothedPotential(self.x,self.y,self.z,epsilon)*self.psi)*self.volume
        

    
    
            
    
    
    
    

    

    
    
    
    
        
    
    



