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
        
        xm,ym,zm = np.meshgrid(xf,yf,zf)
        psi_fine = self.interpolator((xm,ym,zm))
        
        # generate 8 children
        for ichild in range(2):
            for jchild in range(2):
                for kchild in range(2):
                    children[ichild,jchild,kchild] = cell( xf[2*ichild:2*ichild+3], 
                                                    yf[2*jchild:2*jchild+3], zf[2*kchild:2*kchild+3],
                                                    psi_fine[2*ichild:2*ichild+3, 2*jchild:2*jchild+3,2*kchild:2*kchild+3])
        return children

    def checkDivide(self,gradientThreshold):
        try:
            self.grad
        except AttributeError:
            self.gradient_psi()
        
        if np.max(np.abs(self.grad)) > gradientThreshold:
            self.NeedsDividing = True
        else:
            self.NeedsDividing = False

    def computeLaplacian(self):
        try:
            self.grad
        except AttributeError:
            self.gradient_psi()
        
        Dxx = np.gradient(self.grad[0],self.dx,edge_order=2,axis=0)
        Dyy = np.gradient(self.grad[1],self.dy,edge_order=2,axis=1)
        Dzz = np.gradient(self.grad[2],self.dz,edge_order=2,axis=2)
        self.Laplacian = Dxx[1,1,1] + Dyy[1,1,1] + Dzz[1,1,1]  # only use the Laplacian at the midpoint, for now at least

    
    def evaluatePotential(self):
        self.V = potential(self.x[1],self.y[1],self.z[1])
    
    
    
    
            
    
    
    
    

    

    
    
    
    
        
    
    



