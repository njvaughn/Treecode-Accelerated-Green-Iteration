import numpy as np

class Mesh(object):
    '''
    Mesh object.  Will be constructed out of many cells, which will be constructed out of gridpoints
    '''
    def __init__(self, xmin,xmax,nx,ymin,ymax,ny,zmin,zmax,nz):
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
        zvec = np.linspace(zmin,zmax,2*nz+1)
        gridpoints = np.empty((2*nx+1,2*ny+1, 2*nz+1),dtype=object)
        for i in range(2*nx+1):
            for j in range(2*ny+1):
                for k in range(2*nz+1):
                    gridpoints[i,j,k] = GridPt(xvec[i],yvec[j],zvec[k])
        
        # generate cells from the gridpoint objects  
        self.cells = np.empty((nx,ny,nz),dtype=object)
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    self.cells[i,j,k] = Cell( gridpoints[2*i:2*i+3, 2*j:2*j+3, 2*k:2*k+3] )
                    

class Tree(object):
    '''
    Tree object. Constructed of cells, which are composed of gridpoint objects.
    '''
    def __init__(self, xmin,xmax,ymin,ymax,zmin,zmax):
        '''
        Tree constructor:  
        First construct the gridpoints for cell consisting of entire domain.  
        Then construct the cell that are composed of gridpoints. 
        Then construct the root of the tree.
        '''
        # generate gridpoint objects.  
        xvec = np.linspace(xmin,xmax,3)
        yvec = np.linspace(ymin,ymax,3)
        zvec = np.linspace(zmin,zmax,3)
        gridpoints = np.empty((3,3,3),dtype=object)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    gridpoints[i,j,k] = GridPt(xvec[i],yvec[j],zvec[k])
        
        # generate root cell from the gridpoint objects  
        self.cells = np.empty((1,1,1),dtype=object)
        self.cells[0,0,0] = Cell( gridpoints )
        print("Root of tree constructed.")
    
        
        
    def buildTree(self):
#         for cell in self.cells:
        self.cells[0,0,0].recursiveDivide()


class Cell(object):
    '''
    Cell object.  Cells are composed of gridpoint objects.
    '''
    def __init__(self, gridpoints):
        '''
        Cell Constructor.  Cell composed of gridpoint objects
        '''
        self.gridpoints = gridpoints
        
    def getCellBounds(self):
        self.xmin = self.gridpoints[0,0,0].x
        self.xmax = self.gridpoints[2,2,2].x
        self.ymin = self.gridpoints[0,0,0].y
        self.ymax = self.gridpoints[2,2,2].y
        self.zmin = self.gridpoints[0,0,0].z
        self.zmax = self.gridpoints[2,2,2].z
        
        
    def divide(self):
        children = np.empty((2,2,2), dtype=object)
        self.getCellBounds()
        x = np.linspace(self.xmin,self.xmax,5)
        y = np.linspace(self.ymin,self.ymax,5)
        z = np.linspace(self.zmin,self.zmax,5)
        gridpoints = np.empty((5,5,5),dtype=object)
        gridpoints[::2,::2,::2] = self.gridpoints  # the 5x5x5 array of gridpoints should have the original 3x3x3 objects within
        for i in range(5):
            for j in range(5):
                for k in range(5):
                    if gridpoints[i,j,k] == None:
                        gridpoints[i,j,k] = GridPt(x[i],y[j],z[k])

        for i in range(2):
            for j in range(2):
                for k in range(2):
                    children[i,j,k] = Cell(gridpoints[2*i:2*i+3, 2*j:2*j+3, 2*k:2*k+3])
                    
        self.children = children

    def recursiveDivide(self):
        self.getCellBounds()
        if self.xmax - self.xmin > 0.1:
            self.divide()
            for child in self.children:
                child.recursiveDivide()

        
class GridPt(object):
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
    
    def setPsi(self, psi):
        self.psi = psi
        
