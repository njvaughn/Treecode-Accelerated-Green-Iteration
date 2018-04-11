'''
Created on Mar 5, 2018

@author: nathanvaughn
'''
from hydrogenPotential import trueWavefunction
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
        self.setAnalyticPsi(0)  # for now, set the analytic psi value.  Eventually, need to use interpolator
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
        self.testFunctionValue = trueWavefunction(0, self.x,self.y,self.z)**2 +  trueWavefunction(1, self.x,self.y,self.z)**2