'''
Created on Mar 5, 2018

@author: nathanvaughn
'''
from hydrogenPotential import trueWavefunction, potential
import numpy as np

class GridPoint(object):
    '''
    The gridpoint object.  Will contain the coordinates, wavefunction value, and any other metadata such as
    whether or not the wavefunction has been updated, which cells the gridpoint belongs to, etc.
    '''
    def __init__(self, x,y,z):
        '''
        Gridpoint Constructor.  For minimal example, a gridpoint simply has x and y values.
        '''
        self.x = x
        self.y = y
        self.z = z
        self.setAnalyticPsi(1)  # for now, set the analytic psi value.  Eventually, need to use interpolator
#         self.setPsi(np.random.rand(1))  # for now, set the analytic psi value.  Eventually, need to use interpolator
        self.setTestFunctionValue()
        self.finalWavefunction = []
        
    def setPsi(self, psi):
        self.psi = psi
    
    def setAnalyticPsi(self,n):
        self.psi = trueWavefunction(n, self.x,self.y,self.z)
        
    def setTestFunctionValue(self):
        '''
        Set the test function value.  For now, this can be the single atom single electron wavefunction.
        Generally, this should be some representative function that we can use apriori to set up the 
        refined mesh.  Bikash uses single atom densities, or a sum of single atom densities to give an 
        indication of where he should refine before actually computing the many-atom electron density.
        '''
#         self.testFunctionValue = trueWavefunction(0, self.x,self.y,self.z)
        self.testFunctionValue = trueWavefunction(0, self.x,self.y,self.z)**2 +  trueWavefunction(1, self.x,self.y,self.z)**2
#         self.testFunctionValue = trueWavefunction(0, self.x,self.y,self.z)**4 +  trueWavefunction(1, self.x,self.y,self.z)**4

#         epsq = 1e-8
#         r = np.sqrt(self.x**2 + self.y**2 + self.z**2 + epsq)
#         self.testFunctionValue = 1/r**2
#         
#         self.testFunctionValue = trueWavefunction(0, self.x,self.y,self.z)**2 * potential(self.x,self.y,self.z,1e-6)
        
        
        
        
        