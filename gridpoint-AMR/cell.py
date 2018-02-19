import numpy as np
class Cell(object):
    

    def __init__(self, gp00, gp01,gp02,gp10,gp11,gp12,gp20,gp21,gp22):
        '''
        Gridpoint Constructor
        '''
        self.gridpoints = np.empty((3,3),dtype=object)
        self.gridpoints[0,0] = gp00
        self.gridpoints[0,1] = gp01
        self.gridpoints[0,2] = gp02
        self.gridpoints[1,0] = gp10
        self.gridpoints[1,1] = gp11
        self.gridpoints[1,2] = gp12
        self.gridpoints[2,0] = gp20
        self.gridpoints[2,1] = gp21
        self.gridpoints[2,2] = gp22


