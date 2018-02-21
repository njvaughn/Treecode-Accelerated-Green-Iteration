import numpy as np
from scipy.interpolate import RegularGridInterpolator
import itertools
from hydrogenPotential import potential, trueWavefunction

                    

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

        for i, j, k in itertools.product(range(3),range(3),range(3)):
            gridpoints[i,j,k] = GridPoint(xvec[i],yvec[j],zvec[k])
        
        # generate root cell from the gridpoint objects  
#         self.cells = np.empty((1,1,1),dtype=object)
#         self.cells[0,0,0] = Cell( gridpoints )
        self.root = Cell( gridpoints )
        self.root.level = 0
        print("Root of tree constructed.")
            
    def buildTree(self,minLevels,maxLevels,divideTolerance): # call the recursive divison on the root of the tree
        # max depth returns the maximum depth of the tree.  maxLevels is the limit on how large the tree is allowed to be,
        # regardless of division criteria
        def recursiveDivide_fromTree(Cell, minLevels, maxLevels, divideTolerance, maxDepthAchieved=0, currentLevel=0):
            if currentLevel < maxLevels:
                
                if currentLevel < minLevels:
                    Cell.divideFlag = True 
                else:                             
                    Cell.checkIfCellShouldDivide(divideTolerance)
                    
                if Cell.divideFlag == True:   
                    Cell.divide()
                    for i,j,k in itertools.product(range(2),range(2),range(2)):
                        maxDepthAchieved = recursiveDivide_fromTree(Cell.children[i,j,k], minLevels, maxLevels, divideTolerance, maxDepthAchieved, currentLevel+1)
                    
            maxDepthAchieved = max(maxDepthAchieved, currentLevel)                                                                                                                                                       
            return maxDepthAchieved
        
        self.maxDepthAchieved = recursiveDivide_fromTree(self.root, minLevels, maxLevels, divideTolerance, maxDepthAchieved=0, currentLevel=0 )
        
    def walkTree(self, attribute=''):  # walk through the tree, printing out specified data (in this case, the midpoints)
        def recursiveWalkMidpoint(Cell, attribute):  
            midpoint = Cell.gridpoints[1,1,1]
            if hasattr(Cell,attribute):
                print('Level: ', Cell.level,', midpoint: (', midpoint.x,', ',midpoint.y,', ',midpoint.z,'), ', attribute,': ', getattr(Cell,attribute))
            elif hasattr(midpoint,attribute):
                print('Level: ', Cell.level,', midpoint: (', midpoint.x,', ',midpoint.y,', ',midpoint.z,'), ', attribute,': ', getattr(midpoint,attribute))
            else:
                print('Level: ', Cell.level,', midpoint: (', midpoint.x,', ',midpoint.y,', ',midpoint.z,')')
            if hasattr(Cell,'children'):
                print()
                for i,j,k in itertools.product(range(2),range(2),range(2)):
                    recursiveWalkMidpoint(Cell.children[i,j,k],attribute)
                print()
                
        recursiveWalkMidpoint(self.root, attribute)
        


class Cell(object):
    '''
    Cell object.  Cells are composed of gridpoint objects.
    '''
    def __init__(self, gridpoints):
        '''
        Cell Constructor.  Cell composed of gridpoint objects
        '''
        self.gridpoints = gridpoints
        self.getCellBoundsAndVolume()
        
    def getCellBoundsAndVolume(self):
        self.xmin = self.gridpoints[0,0,0].x
        self.xmax = self.gridpoints[2,2,2].x
        self.ymin = self.gridpoints[0,0,0].y
        self.ymax = self.gridpoints[2,2,2].y
        self.zmin = self.gridpoints[0,0,0].z
        self.zmax = self.gridpoints[2,2,2].z
        self.volume = (self.xmax-self.xmin)*(self.ymax-self.ymin)*(self.zmax-self.zmin)
        
    def interpolate_for_division(self):
        if hasattr(self, 'psi'):
            psiCoarse = np.empty((3,3,3))
            xvec = np.array(self.gridpoints[0,0,0].x,self.gridpoints[1,0,0].x,self.gridpoints[2,0,0].x)
            yvec = np.array(self.gridpoints[0,0,0].y,self.gridpoints[0,1,0].y,self.gridpoints[0,2,0].y)
            zvec = np.array(self.gridpoints[0,0,0].z,self.gridpoints[0,0,1].z,self.gridpoints[0,0,2].z)
            for i,j,k in itertools.product(range(3),range(3),range(3)):
                psiCoarse[i,j,k] = self.gridpoints[i,j,k].psi
        
            self.interpolator = RegularGridInterpolator((xvec, yvec, zvec), psiCoarse) 
        else:
            print("Can't generate interpolator because psi hasn't been set yet.")
    
    def checkIfCellShouldDivide(self, divideTolerance):
        # perform midpoint method for integral(testFunction) for parent cell
        # If parent integral differs from the sum of children integrals over the test function
        # then set divideFlag to true.
        self.gridpoints[1,1,1].setTestFunctionValue()
        parentIntegral = self.gridpoints[1,1,1].testFunctionValue*self.volume
        
        childrenIntegral = 0.0
        xmids = np.array([(3*self.xmin+self.xmax)/4, (self.xmin+3*self.xmax)/4])
        ymids = np.array([(3*self.ymin+self.ymax)/4, (self.ymin+3*self.ymax)/4])
        zmids = np.array([(3*self.zmin+self.zmax)/4, (self.zmin+3*self.zmax)/4])
        for i,j,k in itertools.product(range(2),range(2),range(2)):
            tempChild = GridPoint(xmids[i],ymids[j],zmids[k])
            tempChild.setTestFunctionValue()
            childrenIntegral += tempChild.testFunctionValue*(self.volume/8)
        
        if abs(parentIntegral-childrenIntegral) > divideTolerance:
            self.divideFlag = True
        else:
            self.divideFlag = False
        
    def divide(self):
        children = np.empty((2,2,2), dtype=object)
        x = np.linspace(self.xmin,self.xmax,5)
        y = np.linspace(self.ymin,self.ymax,5)
        z = np.linspace(self.zmin,self.zmax,5)
        gridpoints = np.empty((5,5,5),dtype=object)
        gridpoints[::2,::2,::2] = self.gridpoints  # the 5x5x5 array of gridpoints should have the original 3x3x3 objects within
        
        """ 
        Will need interpolator when refining and have a psi estimate on the coarse mesh.
        Don't really need it yet.
        """
#         xm,ym,zm = np.meshgrid(x,y,z, indexing='ij')
#         self.interpolate_for_division() # generate the local interpolator
#         psi_fine = self.interpolator((xm,ym,zm)) # interpolate psi onto the fine mesh
        
        for i, j, k in itertools.product(range(5),range(5),range(5)):
            if gridpoints[i,j,k] == None:
                gridpoints[i,j,k] = GridPoint(x[i],y[j],z[k])
                
        for i, j, k in itertools.product(range(2),range(2),range(2)):
            children[i,j,k] = Cell(gridpoints[2*i:2*i+3, 2*j:2*j+3, 2*k:2*k+3])
            children[i,j,k].parent = self # children should point to their parent
            if hasattr(self,'level'):
                children[i,j,k].level = self.level+1
        
        self.children = children

    
            


        
class GridPoint(object):
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
        self.setAnalyticPsi()  # for now, set the analytic psi value.  Eventually, need to use interpolator
    
    def setPsi(self, psi):
        self.psi = psi
    
    def setAnalyticPsi(self):
        self.psi = trueWavefunction(1, self.x,self.y,self.z)
        
    def setTestFunctionValue(self):
        '''
        Set the test function value.  For now, this can be the single atom single electron wavefunction.
        Generally, this should be some representative function that we can use apriori to set up the 
        refined mesh.  Bikash uses single atom densities, or a sum of single atom densities to give an 
        indication of where he should refine before actually computing the many-atom electron density.
        '''
        self.testFunctionValue = trueWavefunction(1, self.x,self.y,self.z)  
        
class Mesh(object):  
    '''
    Mesh object.  Will be constructed out of many cells, which will be constructed out of gridpoints.
    This will be used if we start with pre-existing mesh and want to set up the cell structure.
    Ideally, we start with a blank slate and can build the tree straight away, never using this mesh object.
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
                    gridpoints[i,j,k] = GridPoint(xvec[i],yvec[j],zvec[k])
        
        # generate cells from the gridpoint objects  
        self.cells = np.empty((nx,ny,nz),dtype=object)
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    self.cells[i,j,k] = Cell( gridpoints[2*i:2*i+3, 2*j:2*j+3, 2*k:2*k+3] )