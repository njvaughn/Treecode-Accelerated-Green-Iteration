from scipy import special as spec
import numpy as np

class Model():    
    
    def __init__(self, model_name):
        self.model_name = model_name
      
    def potential(self,x, D):
        if self.model_name == "Poschl-Teller":
            return -D*(D+1)/2*(1/np.cosh(x))**2
         
    def true_energy(self,n, D):
        if self.model_name == "Poschl-Teller":
            return -(D-n)**2/2
         
    def true_wave(self, n, D, grid):
        if self.model_name == "Poschl-Teller":
            return spec.lpmv(D-n, D, np.tanh(grid))



