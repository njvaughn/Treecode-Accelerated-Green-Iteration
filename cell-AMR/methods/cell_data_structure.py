import numpy as np



class cell(object):
        
    def __init__(self, x,y,z, psi):
        self.psi = psi
        self.x = x
        self.y = y
        self.z = z
        self.dx = self.x[1]-self.x[0]
        self.dy = self.y[1]-self.y[0]
        self.dz = self.z[1]-self.z[0]

    
    def gradient_psi(self):
        self.grad = np.gradient(self.psi,self.dx,self.dy,self.dz,edge_order=2)
    
    
        
    
    



