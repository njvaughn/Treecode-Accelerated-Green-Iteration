'''
Created on Feb 17, 2018

@author: nathanvaughn
'''
import numpy as np


from cell import Cell
from gridpoint import GridPt


class Mesh(object):
    '''
    Mesh object.  Will be constructed out of many cells, which will be constructed out of gridpoints
    '''


#     def __init__(self, xmin,xmax,nx,ymin,ymax,ny,zmin,zmax,nz):
    def __init__(self, xmin,xmax,nx,ymin,ymax,ny):
        '''
        Constructor
        '''
        # generate gridpoint objects.  2n+1 gridpoints if there are n cells in a given dimension
        xvec = np.linspace(xmin,xmax,2*nx+1)
        yvec = np.linspace(ymin,ymax,2*ny+1)
#         print('xvec :', xvec)
#         print('yvec :', yvec)
#         zvec = np.linspace(zmin,zmax,2*nz+1)
#         gridpoints = np.empty((2*nx+1,2*ny+1,2*nz+1),dtype=object)
        gridpoints = np.empty((2*nx+1,2*ny+1),dtype=object)
        for i in range(2*nx+1):
            for j in range(2*ny+1):
                gridpoints[i,j] = GridPt(xvec[i],yvec[j])
#                 for k in range(2*nz+1):
#                     gridpoints[i,j,k] = GridPt(xvec[i],yvec[j],zvec[k])
        
        # generate cells from the gridpoint objects  
#         self.cells = np.empty((nx,ny,nz),dtype=object)
        self.cells = np.empty((nx,ny),dtype=object)
        for i in range(nx):
            for j in range(ny):
                self.cells[i,j] = Cell(gridpoints[2*i,2*j], 
                                       gridpoints[2*i,2*j+1],
                                       gridpoints[2*i,2*j+2],
                                       
                                       gridpoints[2*i+1,2*j],
                                       gridpoints[2*i+1,2*j+1],
                                       gridpoints[2*i+1,2*j+2],
                                       
                                       gridpoints[2*i+2,2*j], 
                                       gridpoints[2*i+2,2*j+1],
                                       gridpoints[2*i+2,2*j+2] )                    
        