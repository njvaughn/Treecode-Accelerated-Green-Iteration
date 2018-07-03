'''
Created on Mar 5, 2018

@author: nathanvaughn
'''
import numpy as np

class GridPoint(object):
    '''
    The gridpoint object for the quadrature points.  Will contain the coordinates, potential values, etc.
    '''
    def __init__(self, x,y,z):
        '''
        Gridpoint Constructor.  For minimal example, a gridpoint simply has x and y values.
        '''
        self.x = x
        self.y = y
        self.z = z
        self.setPhi(np.random.rand(1))  # for now, set the analytic phi value.  Eventually, need to use interpolator
        self.finalWavefunction = []
        
        self.rho = None
        self.V_coulomb = None
        self.V_xc = None
        self.V_ext = None
        self.V_eff = None

    def setExternalPotential(self, atoms):
        self.V_ext = 0.0
        for atom in atoms:
            self.V_ext += atom.V(self.x,self.y,self.z)
            
    def updateVeff(self):
            self.V_eff = self.V_coulomb + self.V_xc + self.V_ext 
           
    def setPhi(self, phi):
        self.phi = phi
    
#     def setAnalyticPhi(self,n):
#         self.phi = trueWavefunction(n, self.x,self.y,self.z)
        
#     def setTestFunctionValue(self):
#         '''
#         Set the test function value.  For now, this can be the single atom single electron wavefunction.
#         Generally, this should be some representative function that we can use apriori to set up the 
#         refined mesh.  Bikash uses single atom densities, or a sum of single atom densities to give an 
#         indication of where he should refine before actually computing the many-atom electron density.
#         '''
# #         self.testFunctionValue = trueWavefunction(0, self.x,self.y,self.z)
#         self.testFunctionValue = trueWavefunction(0, self.x,self.y,self.z)**2
# #         self.testFunctionValue = trueWavefunction(0, self.x,self.y,self.z)**4 +  trueWavefunction(1, self.x,self.y,self.z)**4
# 
# #         epsq = 1e-8
# #         r = np.sqrt(self.x**2 + self.y**2 + self.z**2 + epsq)
# #         self.testFunctionValue = 1/r**2
# #         
# #         self.testFunctionValue = trueWavefunction(0, self.x,self.y,self.z)**2 * potential(self.x,self.y,self.z,1e-6)
        
        
        
        
        