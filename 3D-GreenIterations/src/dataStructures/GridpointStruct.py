'''
Created on Mar 5, 2018

@author: nathanvaughn
'''
import numpy as np

class GridPoint(object):
    '''
    The gridpoint object for the quadrature points.  Will contain the coordinates, potential values, etc.
    '''
    def __init__(self, x,y,z, gaugeShift, atoms, nOrbitals, initPotential=False):
        '''
        Gridpoint Constructor.  For minimal example, a gridpoint simply has x and y values.
        '''
        self.x = x
        self.y = y
        self.z = z
        self.phi = np.zeros(nOrbitals)  # intialize to zero before using the isngle atom data.
        self.gaugeShift = gaugeShift
        self.nOrbital=nOrbitals
        
        self.rho = 0

        
        self.v_hartree = 0.0
        self.v_x = 0.0
        self.v_c = 0.0
        self.v_ext = 0.0
         
#         if initPotential==True:
        self.setExternalPotential(atoms)
        self.updateVeff()

    def setExternalPotential(self, atoms):
        self.v_ext = 0.0
        for atom in atoms:
            self.v_ext += atom.V(self.x,self.y,self.z)
#         self.v_ext += gaugeShift  # add the gauge shift to external potential.
#         self.updateVeff()
        
    def sortOrbitals(self, newOrder):
        tempPhi = np.zeros_like(self.phi)
        for m in range(len(self.phi)):
            tempPhi[m] = self.phi[m]
        
        for m in range(len(self.phi)):
            self.setPhi(tempPhi[newOrder[m]], m)
            
    
            
    def updateVeff(self):
#             # zero out v_hartree and v_xc for testing purposes
#             self.v_hartree = 0.0
#             self.v_xc = 0.0
        self.v_eff = self.v_hartree + self.v_x + self.v_c + self.v_ext + self.gaugeShift # v_gauge
           
    def setPhi(self, phi, orbitalNumber):
        self.phi[orbitalNumber] = phi
        
        
class DensityPoint(object):
    '''
    The gridpoint object for the secondary quadrature points that will be used in the Hartree convolution.  Will contain the coordinates, potential values, etc.
    '''
    def __init__(self, x,y,z):
        '''
        Gridpoint Constructor.  For minimal example, a gridpoint simply has x and y values.
        '''
        self.x = x
        self.y = y
        self.z = z
        self.rho = 0.0
        
    def setRho(self, rho):
        self.rho = rho



    

        
        
        