'''
Created on Mar 5, 2018

@author: nathanvaughn
'''
import numpy as np

class GridPoint(object):
    '''
    The gridpoint object for the quadrature points.  Will contain the coordinates, potential values, etc.
    '''
    def __init__(self, x,y,z, Norbitals):
        '''
        Gridpoint Constructor.  For minimal example, a gridpoint simply has x and y values.
        '''
        self.x = x
        self.y = y
        self.z = z
        self.phi = np.empty(Norbitals)
        for i in range(Norbitals):
            self.setPhi(np.random.rand(1), i)  # for now, set the analytic phi value.  Eventually, need to use interpolator
#         self.finalWavefunction = []
        
#         rands = np.random.rand(5)
        self.rho = 0
#         self.v_coulomb = -rands[1]
#         self.v_xc = -rands[2]
#         self.v_ext = -rands[3]
        
        self.v_coulomb = 0.0
        self.v_xc = 0.0
        self.v_ext = 0.0
        
        self.updateVeff()

    def setExternalPotential(self, atoms):
        self.v_ext = 0.0
        for atom in atoms:
            self.v_ext += atom.V(self.x,self.y,self.z)
        self.updateVeff()
            
    def updateVeff(self):
#             # zero out v_coulomb and v_xc for testing purposes
#             self.v_coulomb = 0.0
#             self.v_xc = 0.0
            self.v_eff = self.v_coulomb + self.v_xc + self.v_ext 
           
    def setPhi(self, phi, orbitalNumber):
        self.phi[orbitalNumber] = phi
    
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
        
        
        
        
        