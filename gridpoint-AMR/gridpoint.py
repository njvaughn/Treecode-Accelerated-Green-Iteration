class GridPt(object):
    '''
    The gridpoint object.  Contains the coordinates, wavefunction value, and any other metadata such as
    whether or not the wavefunction has been updated, which cells the gridpoint belongs to, etc.
    '''
    
#     def __init__(self, x,y,z,psi):
    def __init__(self, x,y):
        '''
        Gridpoint Constructor
        '''
        self.x = x
        self.y = y
#         self.z = z
#         self.psi = psi


