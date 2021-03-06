'''
GridpointStruct.py

Gridpoint structure, initially designed to solve the issue of neighboring cells sharing 
points along a boundary.  If they each point to the same gridpoint struct then they 
can both own and modify the same gridpoint, removing the issue of duplicated points.
However, operating on individial gridpoints doesn't make use of vectorization and
in general is very slow.  

The boundary issue was avoided by using only interior chebyshev points,
although this may have led to more points than absolutely necessary.
'''
import numpy as np

class GridPoint(object):
    '''
    The gridpoint object for the quadrature points.  Will contain the coordinates, potential values, etc.
    '''
    def __init__(self, x,y,z, gaugeShift, atoms, coreRepresentation, nOrbitals, initPotential=False):
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
#         self.setExternalPotential(atoms,coreRepresentation)
        self.updateVeff()

    def setExternalPotential(self, atoms, coreRepresentation):
        self.v_ext = 0.0
        for atom in atoms:
            if coreRepresentation=="AllElectron":
                self.v_ext += atom.V_all_electron(self.x,self.y,self.z)
            elif coreRepresentation=="Pseudopotentail":
                self.v_ext += atom.V_local_pseudopotential(self.x,self.y,self.z)
        
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
        
        
        
        
        