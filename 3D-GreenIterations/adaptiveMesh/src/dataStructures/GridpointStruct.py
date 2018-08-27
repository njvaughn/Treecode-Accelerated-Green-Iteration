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
        self.phi = np.zeros(Norbitals)  # intialize to zero before using the isngle atom data.

        
        self.rho = 0

        
        self.v_coulomb = 0.0
        self.v_x = 0.0
        self.v_c = 0.0
        self.v_ext = 0.0
        
        self.updateVeff()

    def setExternalPotential(self, atoms, gaugeShift):
        self.v_ext = 0.0
        for atom in atoms:
            self.v_ext += atom.V(self.x,self.y,self.z)
        self.v_ext += gaugeShift  # add the gauge shift to external potential.
        self.updateVeff()
            
    def updateVeff(self):
#             # zero out v_coulomb and v_xc for testing purposes
#             self.v_coulomb = 0.0
#             self.v_xc = 0.0
            self.v_eff = self.v_coulomb + self.v_x + self.v_c + self.v_ext # v_gauge
           
    def setPhi(self, phi, orbitalNumber):
        self.phi[orbitalNumber] = phi
    

        
        
        