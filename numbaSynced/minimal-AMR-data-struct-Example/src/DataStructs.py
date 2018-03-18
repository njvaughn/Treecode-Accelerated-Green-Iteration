import numpy as np

class Mesh(object):
    '''
    Mesh object.  Will be constructed out of many cells, which will be constructed out of gridpoints
    '''
    def __init__(self, xmin,xmax,nx,ymin,ymax,ny):
        '''
        Mesh constructor:  
        First construct the gridpoints.  
        Then construct the cells that are composed of gridpoints. 
        Then construct the mesh as a 2D or 3D array of cells.
        '''
        # generate gridpoint objects.  2n+1 gridpoints if there are n cells in a given dimension.  
        # These will sit together in memory, and Cell objects will point to 9 of them (for 2D).
        xvec = np.linspace(xmin,xmax,2*nx+1)
        yvec = np.linspace(ymin,ymax,2*ny+1)
        gridpoints = np.empty((2*nx+1,2*ny+1),dtype=object)
        for i in range(2*nx+1):
            for j in range(2*ny+1):
                gridpoints[i,j] = GridPt(xvec[i],yvec[j])
        
        # generate cells from the gridpoint objects  
        self.cells = np.empty((nx,ny),dtype=object)
        for i in range(nx):
            for j in range(ny):
                self.cells[i,j] = Cell( # left column of gridpoints
                                       gridpoints[2*i,2*j], 
                                       gridpoints[2*i,2*j+1],
                                       gridpoints[2*i,2*j+2],
                                        # middle column of gridpoints
                                       gridpoints[2*i+1,2*j],
                                       gridpoints[2*i+1,2*j+1],
                                       gridpoints[2*i+1,2*j+2],
                                        # right column of gridpoints
                                       gridpoints[2*i+2,2*j], 
                                       gridpoints[2*i+2,2*j+1],
                                       gridpoints[2*i+2,2*j+2] )

class Cell(object):
    '''
    Cell object.  Cells are composed of gridpoint objects.
    '''
    def __init__(self, gp00, gp01,gp02,gp10,gp11,gp12,gp20,gp21,gp22):
        '''
        Cell Constructor.  Cell composed of gridpoint objects
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
        
class GridPt(object):
    '''
    The gridpoint object.  Will contain the coordinates, wavefunction value, and any other metadata such as
    whether or not the wavefunction has been updated, which cells the gridpoint belongs to, etc.
    '''
    def __init__(self, x,y):
        '''
        Gridpoint Constructor.  For minimal example, a gridpoint simply has x and y values.
        '''
        self.x = x
        self.y = y
