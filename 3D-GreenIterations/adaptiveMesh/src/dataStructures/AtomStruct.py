'''
@author: nathanvaughn
'''
import numpy as np

class Atom(object):
    '''
    The gridpoint object.  Will contain the coordinates, wavefunction value, and any other metadata such as
    whether or not the wavefunction has been updated, which cells the gridpoint belongs to, etc.
    '''
    def __init__(self, x,y,z,atomicNumber):
        '''
        Atom Constructor
        '''
        self.x = x
        self.y = y
        self.z = z
        self.atomicNumber = atomicNumber
       
    def V(self,x,y,z):
        r = np.sqrt( (x - self.x)**2 + (y-self.y)**2 + (z-self.z)**2)
        if r ==0.0:
            return 0.0
        return -self.atomicNumber/r
        
        
        
        