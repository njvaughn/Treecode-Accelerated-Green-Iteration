import numpy as np
from cellDataStructure import cell
from hydrogenPotential import trueWavefunction




class Grid(object):
    '''
    Given domain parameters, generate a uniform (coarse) structure of cells
    '''
    
    
    def __init__(self, xmin, xmax, ymin, ymax, zmin, zmax, nx, ny, nz):
        
        """
        Initialize a 3D array of cells containing the analytic psi (for testing)
        """
        self.xgrid = np.linspace(xmin,xmax,2*nx+1)
        self.ygrid = np.linspace(ymin,ymax,2*ny+1)
        self.zgrid = np.linspace(zmin,zmax,2*nz+1)
        
#         self.cells = np.empty((nx,ny,nz),dtype=object)
        self.cells = np.empty((nx*ny*nz),dtype=object)
        for i in range(nx):
            x = self.xgrid[i*2:i*2+3]
            for j in range(ny):
                y = self.ygrid[j*2:j*2+3]
                for k in range(nz):
                    z = self.zgrid[k*2:k*2+3]
                    xm,ym,zm = np.meshgrid(x,y,z,indexing='ij')
                    psi = trueWavefunction(1, xm, ym, zm)
                    index = ny*nz*i + nz*j + k
                    self.cells[index] = cell(x,y,z,psi)

    
    def GridRefinement(self,psiGradientThreshold):
        count = 0
        numcells = len(self.cells)
        children_array = np.array([])
        for index in range(numcells):  # note, after one level of refinement won't know the shape of cells necessarily
#             print(index)
            self.cells[index].checkDivide(psiGradientThreshold)
            if self.cells[index].NeedsDividing == True:
                count+=1           
                children = self.cells[index].divide().flatten()     
                children_array = np.concatenate([children_array, children])
        print('dividing ', count,' cells')
        indices_for_deleting = []
        for index in range(numcells):
            if self.cells[index].NeedsDividing == True:
                indices_for_deleting.append(index)
        self.cells = np.delete(self.cells,indices_for_deleting)
        self.cells = np.concatenate([self.cells, children_array])
    
    def extractGridFromCellArray(self):
        x = np.array([])
        y = np.array([])
        z = np.array([])
        psi = np.array([])
        for index in range(len(self.cells)):
            x = np.append(x,self.cells[index].x[1])
            y = np.append(y,self.cells[index].y[1])
            z = np.append(z,self.cells[index].z[1])
            psi = np.append(psi, self.cells[index].psi[1,1,1])
            
        return x,y,z,psi
    
    
                
            
    
    
  
        
        
    
    



